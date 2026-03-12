# attacks.py
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import torch
import torch.nn.functional as F

# ============================== Config ==============================

@dataclass
class AttackConfig:
    enabled: bool = False
    # one client can run one or more modes at once (combined):
    # Supported modes: "psmu", "ahum", "random", "bandwagon", "hard", "demote"
    modes: List[str] = None
    # keep legacy "kind" for backward-compat (used if modes is None)
    kind: str = "psmu"
    # selection is done in server; these are just shared params:
    frac: float = 0.05               # only used by server
    target_item: int = 1
    topk_hot: int = 100
    alt_k: int = 8
    K_top: int = 40
    seq_len: int = 30
    num_pseudo: int = 4
    beta: float = 0.1                # global scale for poison loss
    # optional per-mode weights when combined
    weights: Optional[Dict[str, float]] = None  # e.g., {"psmu":1.0, "hard":0.5}


# ============================ Base Strategy ============================

class AttackStrategy:
    """
    Handles single-mode strategy.
    Supports: 'psmu', 'ahum', 'random', 'bandwagon', 'hard', 'demote'.
    """
    def __init__(self, cfg: AttackConfig, items_popularity) -> None:
        self.cfg = cfg
        self.mode = (cfg.kind or "psmu").lower()
        
        # accept list/np.ndarray; ensure 1D float array
        pop = np.asarray(items_popularity).astype(float).ravel() if items_popularity is not None else np.array([0.0])
        if pop.size == 0: pop = np.array([0.0])
        self.items_popularity = pop
        
        # Store num_items for Random attack
        self.num_items = len(pop) - 1 

        self._precompute_sets()

    # ---------- precompute hot/alt ----------
    def _precompute_sets(self) -> None:
        pop = self.items_popularity
        n_items = len(pop) - 1  # excluding pad=0
        if n_items <= 0:
            self.hot_items = []
            self.alt_items = []
            return

        # clamp target id into valid range [1..n_items]
        self.cfg.target_item = int(max(1, min(int(self.cfg.target_item), n_items)))

        # top hot items by popularity (skip padding idx 0)
        # REUSED BY: PSMU, A-hum, Bandwagon
        k = max(1, min(int(self.cfg.topk_hot), n_items))
        hot = np.argsort(pop[1:])[::-1][:k] + 1
        self.hot_items = hot.tolist()

        # alternatives for PSMU: popular items except target
        alt_k = max(0, int(self.cfg.alt_k))
        self.alt_items = [i for i in self.hot_items if i != self.cfg.target_item][:alt_k]

    # ---------- public API ----------
    def augment_sequences(self, local_sequences: List[List[int]], model) -> List[List[int]]:
        """
        Inject fake data. 
        - 'psmu', 'random', 'bandwagon': Inject fake sequences.
        - 'ahum', 'hard', 'demote': Usually do NOT inject data (Model Poisoning), return empty list.
        """
        if self.mode == "psmu":
            syn = self._build_psmu_sequences(model)
        elif self.mode == "random":
            syn = self._build_random_sequences()
        elif self.mode == "bandwagon":
            syn = self._build_bandwagon_sequences()
        elif self.mode == "ahum":
            # Original A-hum implementation logic creates pseudo sequences for gradient alignment
            syn = self._build_ahum_sequences(model)
        else:
            # "hard" and "demote" attacks typically manipulate gradients directly, no extra data needed
            syn = []
            
        return local_sequences + syn

    def poison_loss(self, seq_output: torch.Tensor, model) -> torch.Tensor:
        """
        Compute auxiliary loss.
        - 'psmu', 'ahum', 'hard', 'demote': Return specific loss.
        - 'random', 'bandwagon': Return 0 (Data poisoning only).
        """
        if self.mode == "psmu":
            return self._psmu_loss(seq_output, model)
        elif self.mode == "ahum":
            return self._ahum_loss(seq_output, model)
        elif self.mode == "hard":
            return self._hard_loss(seq_output, model)
        elif self.mode == "demote":
            return self._demote_loss(seq_output, model)
        else:
            # Random/Bandwagon have no extra loss term
            return seq_output.new_zeros(())

    # ---------------- 1. PSMU (Promotion) ----------------
    def _build_psmu_sequences(self, model) -> List[List[int]]:
        L = int(self.cfg.seq_len)
        if L <= 0: return []
        seqs: List[List[int]] = []
        for _ in range(int(self.cfg.num_pseudo)):
            # Reuse _sample_pop_items logic
            base_len = max(1, L - (2 + len(self.alt_items)))
            base = self._sample_pop_items(base_len, exclude={self.cfg.target_item, *self.alt_items})
            seq: List[int] = []
            for b in base:
                seq.append(int(b))
                if np.random.rand() < 0.15:
                    seq.append(int(self.cfg.target_item))
                if np.random.rand() < 0.15 and self.alt_items:
                    seq.append(int(np.random.choice(self.alt_items)))
            # ensure presence
            seq.extend([int(self.cfg.target_item), int(self.cfg.target_item)])
            if self.alt_items:
                a_take = min(4, len(self.alt_items))
                seq.extend([int(a) for a in np.random.choice(self.alt_items, size=a_take, replace=False).tolist()])
            seqs.append(seq[-L:])
        return seqs

    def _psmu_loss(self, seq_output: torch.Tensor, model) -> torch.Tensor:
        device = seq_output.device
        B, H = seq_output.shape
        tgt_id = int(self.cfg.target_item)
        tgt = torch.tensor([tgt_id], device=device)
        e_tgt = self._item_embedding(model, tgt)           # [1,H]
        s_tgt = (seq_output * e_tgt.expand(B, -1)).sum(-1) # [B]
        
        # Loss: Maximize target score
        loss = F.binary_cross_entropy_with_logits(s_tgt, torch.ones_like(s_tgt))

        # Loss: Suppress alternatives
        if len(self.alt_items) > 0:
            alt = torch.tensor(self.alt_items, device=device)
            e_alt = self._item_embedding(model, alt)       # [M,H]
            s_alt = seq_output @ e_alt.T                   # [B,M]
            zeros = torch.zeros_like(s_alt)
            loss = loss + F.binary_cross_entropy_with_logits(s_alt, zeros).mean()

        return loss

    # ---------------- 2. A-hum (Camouflage) ----------------
    def _build_ahum_sequences(self, model) -> List[List[int]]:
        L, K = int(self.cfg.seq_len), int(self.cfg.K_top)
        if L <= 0 or K <= 0: return []
        emb = self._item_embedding_weight(model)           # [V,H]
        if emb is None or emb.shape[0] <= 1: return []
        with torch.no_grad():
            V = emb[1:].detach().cpu().numpy()             # drop pad
        _, H = V.shape
        seqs: List[List[int]] = []
        for _ in range(int(self.cfg.num_pseudo)):
            u = np.random.randn(H).astype(np.float32)
            u /= (np.linalg.norm(u) + 1e-8)
            topK = np.argsort(V @ u)[-min(K, V.shape[0]):] + 1
            seqs.append(np.random.choice(topK, size=L, replace=True).tolist())
        return seqs

    def _ahum_loss(self, seq_output: torch.Tensor, model) -> torch.Tensor:
        device = seq_output.device
        # REUSE: self.hot_items used for center calculation
        if len(self.hot_items) == 0:
            return seq_output.new_zeros(())

        tgt_id = int(self.cfg.target_item)
        hot = torch.tensor(self.hot_items, device=device)
        with torch.no_grad():
            e_center = self._item_embedding(model, hot).mean(dim=0)  # [H] (detached)
        e_weight = self._item_embedding_weight(model)
        if e_weight is None or tgt_id >= e_weight.shape[0]:
            return seq_output.new_zeros(())
        e_target_param = e_weight[tgt_id]                    # [H]
        return F.mse_loss(e_target_param, e_center)

    # ---------------- 3. Random Attack (Data Poisoning) ----------------
    def _build_random_sequences(self) -> List[List[int]]:
        """
        Fills sequence with completely random items + Target at the end.
        """
        L = int(self.cfg.seq_len)
        if L <= 0 or self.num_items <= 0: return []
        seqs: List[List[int]] = []
        for _ in range(int(self.cfg.num_pseudo)):
            # Random items from 1 to num_items
            rand_items = np.random.randint(1, self.num_items + 1, size=L-1).tolist()
            # Target at end
            rand_items.append(int(self.cfg.target_item))
            seqs.append(rand_items)
        return seqs

    # ---------------- 4. Bandwagon Attack (Data Poisoning) ----------------
    def _build_bandwagon_sequences(self) -> List[List[int]]:
        """
        Fills sequence with Top-K Popular items + Target at the end.
        REUSES: self.hot_items
        """
        L = int(self.cfg.seq_len)
        if L <= 0 or not self.hot_items: return []
        seqs: List[List[int]] = []
        for _ in range(int(self.cfg.num_pseudo)):
            # Sample only from hot items
            hot_part = np.random.choice(self.hot_items, size=L-1, replace=True).tolist()
            hot_part.append(int(self.cfg.target_item))
            seqs.append(hot_part)
        return seqs

    # ---------------- 5. Hard Attack (Promotion) ----------------
    def _hard_loss(self, seq_output: torch.Tensor, model) -> torch.Tensor:
        """
        Directly maximizes Target Item score without camouflage.
        """
        device = seq_output.device
        B, H = seq_output.shape
        tgt_id = int(self.cfg.target_item)
        tgt = torch.tensor([tgt_id], device=device)
        
        e_tgt = self._item_embedding(model, tgt)           # [1,H]
        s_tgt = (seq_output * e_tgt.expand(B, -1)).sum(-1) # [B]
        
        # Simple Maximization: Minimize BCE(TargetScore, 1.0)
        loss = F.binary_cross_entropy_with_logits(s_tgt, torch.ones_like(s_tgt))
        return loss

    # ---------------- 6. Demote Attack (Suppression) ----------------
    def _demote_loss(self, seq_output: torch.Tensor, model) -> torch.Tensor:
        """
        Directly minimizes Target Item score.
        Goal: Make the model predict label 0 for this item.
        """
        device = seq_output.device
        B, H = seq_output.shape
        tgt_id = int(self.cfg.target_item)
        tgt = torch.tensor([tgt_id], device=device)
        
        e_tgt = self._item_embedding(model, tgt)           # [1,H]
        s_tgt = (seq_output * e_tgt.expand(B, -1)).sum(-1) # [B]
        
        # Simple Minimization: Minimize BCE(TargetScore, 0.0)
        loss = F.binary_cross_entropy_with_logits(s_tgt, torch.zeros_like(s_tgt))
        return loss

    # -------------------- utils --------------------
    def _sample_pop_items(self, n: int, exclude: Optional[set] = None) -> List[int]:
        if n <= 0: return []
        pop = self.items_popularity
        ids = np.arange(len(pop))
        mask = np.ones_like(pop, dtype=bool)
        mask[0] = False  # drop padding
        if exclude:
            for x in exclude:
                if 0 <= int(x) < len(mask):
                    mask[int(x)] = False
        pool = ids[mask]
        if pool.size == 0: return []
        w = pop[mask].astype(float)
        w = np.maximum(w, 0.0) + 1e-6
        w = w / w.sum()
        replace = n > pool.size
        return np.random.choice(pool, size=n, replace=replace, p=w).astype(int).tolist()

    def _item_embedding(self, model, idx: torch.Tensor) -> torch.Tensor:
        emb = getattr(model, "item_embedding", None) or getattr(model, "item_emb", None)
        if emb is None:
            raise AttributeError("Model has no 'item_embedding' or 'item_emb'.")
        return emb(idx)

    def _item_embedding_weight(self, model) -> Optional[torch.Tensor]:
        emb = getattr(model, "item_embedding", None) or getattr(model, "item_emb", None)
        return None if emb is None else emb.weight


# ===================== Combined (both on same client) =====================

class CombinedAttackStrategy:
    def __init__(self, strategies: List[AttackStrategy], weights: Optional[Dict[str, float]] = None) -> None:
        assert len(strategies) > 0
        self.strategies = strategies
        self.weights = weights or {}

    @property
    def cfg(self) -> AttackConfig:
        return self.strategies[0].cfg

    def augment_sequences(self, local_sequences: List[List[int]], model) -> List[List[int]]:
        out = list(local_sequences)
        for s in self.strategies:
            out = s.augment_sequences(out, model)
        return out

    def poison_loss(self, seq_output: torch.Tensor, model) -> torch.Tensor:
        total = seq_output.new_zeros(())
        for s in self.strategies:
            w = float(self.weights.get(s.mode, 1.0))
            total = total + w * s.poison_loss(seq_output, model)
        return total


# ========================= Builders (factories) =========================

def build_attack_strategy(master_cfg: dict, items_popularity) -> Optional[object]:
    if not master_cfg or not master_cfg.get("enabled", False):
        return None

    modes = master_cfg.get("modes")
    if not modes:
        modes = [master_cfg.get("kind", "psmu")]

    strategies: List[AttackStrategy] = []
    for mode in modes:
        cfg = AttackConfig(
            enabled=True,
            modes=[mode],
            kind=str(mode),
            frac=float(master_cfg.get("frac", 0.05)),
            target_item=int(master_cfg["target_item"]),
            topk_hot=int(master_cfg.get("topk_hot", 100)),
            alt_k=int(master_cfg.get("alt_k", 8)),
            K_top=int(master_cfg.get("K_top", 40)),
            seq_len=int(master_cfg.get("seq_len", 30)),
            num_pseudo=int(master_cfg.get("num_pseudo", 4)),
            beta=float(master_cfg.get("beta", 0.1)),
            weights=master_cfg.get("weights"),
        )
        strategies.append(AttackStrategy(cfg, items_popularity))

    if len(strategies) == 1:
        return strategies[0]
    else:
        return CombinedAttackStrategy(strategies, weights=master_cfg.get("weights", None))


def build_attack_strategies(master_cfg: dict, items_popularity) -> Dict[str, AttackStrategy]:
    if not master_cfg or not master_cfg.get("enabled", False):
        return {}
    modes = master_cfg.get("modes", [master_cfg.get("kind", "psmu")])
    out: Dict[str, AttackStrategy] = {}
    for mode in modes:
        cfg = AttackConfig(
            enabled=True,
            modes=[mode],
            kind=str(mode),
            frac=float(master_cfg.get("frac", 0.05)),
            target_item=int(master_cfg["target_item"]),
            topk_hot=int(master_cfg.get("topk_hot", 100)),
            alt_k=int(master_cfg.get("alt_k", 8)),
            K_top=int(master_cfg.get("K_top", 40)),
            seq_len=int(master_cfg.get("seq_len", 30)),
            num_pseudo=int(master_cfg.get("num_pseudo", 4)),
            beta=float(master_cfg.get("beta", 0.1)),
            weights=master_cfg.get("weights"),
        )
        out[mode] = AttackStrategy(cfg, items_popularity)
    return out