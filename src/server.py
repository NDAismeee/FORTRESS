import os
import random
import copy
import numpy as np
import torch
import gc
import time
import io

from logger import CSVLogger
from client import Client
from model.sasrec import SASRec
from evaluate import evaluate, evaluate_er5
from attacks import build_attack_strategy, build_attack_strategies


class Server:
    def __init__(self, global_model_config, train_config, clients_data, device):
        self.device = device
        self.train_config = train_config
        self.clients_data = clients_data  # raw data only
        self.global_model_config = global_model_config
        self.global_model = SASRec(**global_model_config).to(device)
        self.global_weights = self.global_model.state_dict()

        # Expanded CSV schema for efficiency experiments
        self.logger = CSVLogger(
            log_dir="logs",
            fieldnames=[
                # identity
                "round", "phase",

                # metrics
                "loss", "HR@20", "NDCG@20", "Recall@20", "ER@5",

                # participation
                "num_selected", "num_trained", "num_malicious",

                # communication (bytes)
                "bytes_down_per_client",
                "bytes_up_avg_per_client",
                "bytes_up_total_round",
                "bytes_round_total",
                "cum_bytes_total",

                # time (seconds)
                "time_round_sec",
                "time_clients_sec",
                "time_agg_sec",
                "time_defense_sec",
                "time_eval_sec",
                "cum_time_sec",
            ],
        )

        os.makedirs(self.train_config.get("save_dir", "checkpoints"), exist_ok=True)

        # ---- Attack orchestration ----
        self.attack_master = self.train_config.get("attack", {"enabled": False})
        self.attack_enabled = bool(self.attack_master.get("enabled", False))
        self.items_popularity = np.array(self.train_config.get("items_popularity", []))

        # If you want BOTH on the same malicious client, use combined_attack
        self.combined_attack = None
        # If you want to assign different attacks to different clients, use strategies
        self.attack_strategies = {}

        if self.attack_enabled:
            # Combined (same client runs both), if master config has multiple modes
            self.combined_attack = build_attack_strategy(self.attack_master, self.items_popularity)
            # Per-mode (server can split malicious clients across modes)
            self.attack_strategies = build_attack_strategies(self.attack_master, self.items_popularity)

    def aggregate(self, client_weights):
        # client_weights are on CPU
        if not client_weights:
            return self.global_weights  # keep old weights if no updates

        avg_weights = copy.deepcopy(client_weights[0])
        for key in avg_weights:
            for i in range(1, len(client_weights)):
                avg_weights[key] += client_weights[i][key]
            avg_weights[key] /= len(client_weights)
        return avg_weights

    def _sizeof_payload_bytes(self, obj) -> int:
        """Serialize object to measure real communication payload size (bytes)."""
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        return buffer.getbuffer().nbytes

    # ---------------- server-side defense ----------------
    def apply_server_defense(self):
        """
        Apply popularity-aware defense (Lsep + Lvar) after aggregation.
        """
        if not self.attack_enabled or "target_item" not in self.attack_master:
            return

        tgt = int(self.attack_master["target_item"])
        k_hot = int(self.attack_master.get("topk_hot", 100))
        pop = self.items_popularity

        if tgt <= 0 or tgt >= len(self.global_model.item_embedding.weight):
            return  # invalid target

        # 1. Identify top-K popular items (excluding padding idx 0)
        if len(pop) > 1:
            hot_items = np.argsort(pop[1:])[::-1][:k_hot] + 1
            hot_items = torch.tensor(hot_items, device=self.device)
        else:
            return

        emb = self.global_model.item_embedding.weight  # [V, H]

        # 2. Separation loss (Lsep): suppress similarity target vs hot
        e_tgt = emb[tgt]        # [H]
        e_hot = emb[hot_items]  # [K,H]
        sep_loss = torch.relu((e_tgt @ e_hot.T).mean())

        # 3. Variance loss (Lvar): encourage diversity in target embedding
        var_loss = -torch.var(e_tgt, unbiased=False)

        lam_sep = float(self.train_config.get("lambda_sep", 0.0))
        lam_var = float(self.train_config.get("lambda_var", 0.0))
        total_loss = lam_sep * sep_loss + lam_var * var_loss

        if total_loss.item() != 0.0:
            optimizer = torch.optim.SGD([emb], lr=1e-3)  # tiny lr
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(f"[Server Defense] Applied (loss={total_loss.item():.4f})")

    # ---------------- internal helpers ----------------
    def _choose_malicious_ids(self, selected_ids):
        """Return set of malicious client ids based on num_malicious or frac."""
        if not self.attack_enabled:
            return set()

        if "num_malicious" in self.attack_master:
            mal_count = int(self.attack_master["num_malicious"])
        else:
            frac = float(self.attack_master.get("frac", 0.05))
            mal_count = max(1, int(round(frac * len(selected_ids))))
        mal_count = min(mal_count, len(selected_ids))
        return set(random.sample(selected_ids, mal_count)) if mal_count > 0 else set()

    def _assign_attacks_to_clients(self, selected_ids):
        """
        Returns:
          attack_map: dict[cid] -> attack object (None for benign)
          malicious_ids: set of malicious client ids
        """
        attack_map = {}
        malicious_ids = self._choose_malicious_ids(selected_ids)
        if not self.attack_enabled or not malicious_ids:
            return attack_map, set()

        modes = list(self.attack_strategies.keys())  # e.g., ["psmu"] or ["psmu","ahum"]
        combine_flag = bool(self.attack_master.get("combine_on_client", False)) or len(modes) <= 1

        if combine_flag:
            for cid in malicious_ids:
                attack_map[cid] = self.combined_attack
            return attack_map, malicious_ids

        # Otherwise, split across modes according to optional weights
        split = self.attack_master.get("split", {})  # {"psmu":0.6, "ahum":0.4}
        weights = [float(split.get(m, 1.0)) for m in modes]
        total_w = sum(weights) if sum(weights) > 0 else 1.0
        weights = [w / total_w for w in weights]

        mal_list = list(malicious_ids)
        mal_count = len(mal_list)

        quotas = [int(round(mal_count * w)) for w in weights]
        while sum(quotas) > mal_count:
            i = quotas.index(max(quotas))
            quotas[i] -= 1
        while sum(quotas) < mal_count:
            i = quotas.index(min(quotas))
            quotas[i] += 1

        start = 0
        for mode, q in zip(modes, quotas):
            for cid in mal_list[start:start + q]:
                attack_map[cid] = self.attack_strategies[mode]
            start += q

        return attack_map, malicious_ids

    def federated_train(self, num_rounds, clients_per_round, user_seqs, test_seqs, itemnum):
        num_clients = len(self.clients_data)
        best_hr = 0
        save_dir = self.train_config.get("save_dir", "checkpoints")

        # cumulative stats for efficiency experiments
        cum_bytes_total = 0
        cum_time_sec = 0.0

        eval_every = int(self.train_config.get("eval_every", 1))
        top_k = int(self.train_config["top_k"])

        for rnd in range(1, num_rounds + 1):
            print(f"\n--- Round {rnd} ---")

            t_round0 = time.time()

            selected_ids = random.sample(range(num_clients), clients_per_round)

            # Decide malicious and assign attacks for this round
            attack_map, malicious_ids = self._assign_attacks_to_clients(selected_ids)
            num_malicious = len(malicious_ids) if self.attack_enabled else 0

            # Server -> client payload (broadcast global weights)
            bytes_down_per_client = self._sizeof_payload_bytes(self.global_weights)

            client_updates = []
            round_losses = []

            # per-round comm stats
            bytes_up_total_round = 0
            num_trained = 0

            # time: clients local training (server-side measured)
            t_clients0 = time.time()

            for cid in selected_ids:
                attack_for_client = attack_map.get(cid, None)

                client = Client(
                    client_id=cid,
                    local_data=self.clients_data[cid],
                    model_config=self.global_model_config,
                    train_config=self.train_config,
                    device=self.device,
                    attack=attack_for_client,
                )
                client.set_weights(self.global_weights)

                updated_weights, client_loss = client.train()  # updated_weights on CPU

                # Tag for debug
                tag = "OK"
                if attack_for_client:
                    if hasattr(attack_for_client, "strategies"):
                        modes = [s.mode for s in attack_for_client.strategies]
                        tag = "+".join(m.upper() for m in modes)
                    else:
                        tag = attack_for_client.mode.upper()

                # If loss=0.0 or None => skipped
                if client_loss is None or client_loss == 0.0:
                    print(f"[{tag}] Client {cid} | SKIPPED (Not enough data)")
                else:
                    print(f"[{tag}] Client {cid} | Local Loss: {client_loss:.4f}")
                    client_updates.append(updated_weights)
                    round_losses.append(client_loss)

                    num_trained += 1
                    bytes_up_total_round += self._sizeof_payload_bytes(updated_weights)

                # Cleanup
                del client
                del updated_weights
                gc.collect()
                torch.cuda.empty_cache()

            time_clients_sec = time.time() - t_clients0

            # aggregation time
            t_agg0 = time.time()
            if client_updates:
                self.global_weights = self.aggregate(client_updates)
                self.global_model.load_state_dict(self.global_weights)
            else:
                print(f"[Warning] Round {rnd} skipped: No clients had enough data to train.")
            time_agg_sec = time.time() - t_agg0

            # defense time
            t_def0 = time.time()
            if client_updates:
                self.apply_server_defense()
            time_defense_sec = time.time() - t_def0

            # Save model after every round (kept as-is)
            torch.save(self.global_model.state_dict(), os.path.join(save_dir, "model.pt"))

            # evaluation (optional)
            metrics = None
            er_value = None
            time_eval_sec = 0.0

            if rnd % eval_every == 0:
                t_eval0 = time.time()

                metrics = evaluate(
                    self.global_model, user_seqs, test_seqs, itemnum, self.device, top_k=top_k
                )
                print(f"Evaluation @ Round {rnd}: {metrics}")

                # ER@5 if attacks enabled
                if self.attack_enabled and "target_item" in self.attack_master:
                    tgt = int(self.attack_master["target_item"])
                    er = evaluate_er5(self.global_model, user_seqs, itemnum, self.device, tgt, top_k=5)
                    er_value = er["ER@5"] if isinstance(er, dict) else er
                    print(f"Target {tgt} Exposure (ER@5): {er_value}")

                time_eval_sec = time.time() - t_eval0

            # compute comm aggregates
            bytes_up_avg_per_client = bytes_up_total_round / max(1, num_trained)
            bytes_round_total = (len(selected_ids) * bytes_down_per_client) + bytes_up_total_round
            cum_bytes_total += bytes_round_total

            # compute time aggregates
            time_round_sec = time.time() - t_round0
            cum_time_sec += time_round_sec

            # mean train loss (only successful clients)
            mean_train_loss = float(np.mean(round_losses)) if round_losses else 0.0

            # ---- LOG train row (EVERY round) ----
            self.logger.log({
                "round": rnd,
                "phase": "train",
                "loss": mean_train_loss,
                "HR@20": None,
                "NDCG@20": None,
                "Recall@20": None,
                "ER@5": None,

                "num_selected": len(selected_ids),
                "num_trained": num_trained,
                "num_malicious": num_malicious,

                "bytes_down_per_client": bytes_down_per_client,
                "bytes_up_avg_per_client": bytes_up_avg_per_client,
                "bytes_up_total_round": bytes_up_total_round,
                "bytes_round_total": bytes_round_total,
                "cum_bytes_total": cum_bytes_total,

                "time_round_sec": time_round_sec,
                "time_clients_sec": time_clients_sec,
                "time_agg_sec": time_agg_sec,
                "time_defense_sec": time_defense_sec,
                "time_eval_sec": time_eval_sec,
                "cum_time_sec": cum_time_sec,
            })

            # ---- LOG eval row (ONLY when evaluated) ----
            if metrics is not None:
                self.logger.log({
                    "round": rnd,
                    "phase": "eval",
                    "loss": None,
                    "HR@20": metrics.get("HR@20"),
                    "NDCG@20": metrics.get("NDCG@20"),
                    "Recall@20": metrics.get("Recall@20"),
                    "ER@5": er_value,

                    "num_selected": len(selected_ids),
                    "num_trained": num_trained,
                    "num_malicious": num_malicious,

                    "bytes_down_per_client": bytes_down_per_client,
                    "bytes_up_avg_per_client": bytes_up_avg_per_client,
                    "bytes_up_total_round": bytes_up_total_round,
                    "bytes_round_total": bytes_round_total,
                    "cum_bytes_total": cum_bytes_total,

                    "time_round_sec": time_round_sec,
                    "time_clients_sec": time_clients_sec,
                    "time_agg_sec": time_agg_sec,
                    "time_defense_sec": time_defense_sec,
                    "time_eval_sec": time_eval_sec,
                    "cum_time_sec": cum_time_sec,
                })

                # Save best model based on HR@top_k
                key_hr = f"HR@{top_k}"
                if key_hr in metrics and metrics[key_hr] > best_hr:
                    best_hr = metrics[key_hr]
                    torch.save(self.global_model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                    print("Best model updated.")
