import numpy as np
import torch
import random
from typing import Iterable, Union, Sequence

def _get_item_embedding_layer(model):
    emb = getattr(model, "item_embedding", None) or getattr(model, "item_emb", None)
    if emb is None:
        raise AttributeError("Model must expose `item_embedding` (or `item_emb`).")
    return emb

def _first_item(x: Union[int, Sequence[int], torch.Tensor]) -> int:
    if isinstance(x, torch.Tensor):
        x = x.view(-1)
        return int(x[0].item())
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return int(x[0])
    return int(x)

def evaluate(model, user_seqs: Iterable[Iterable[int]], test_seqs: Iterable[Iterable[int]],
             itemnum: int, device: torch.device, top_k: int = 20, num_neg: int = 100):
    """
    Standard leave-one-out eval:
      - For each user: score GT item vs. `num_neg` random negatives
      - Return HR@K and NDCG@K
    """
    model.eval()
    HR, NDCG, REC = [], [], []
    max_len = getattr(model, "max_seq_len", 50)
    item_emb = _get_item_embedding_layer(model)

    with torch.no_grad():
        for user_seq, gt_item in zip(user_seqs, test_seqs):
            if not user_seq or gt_item is None or (isinstance(gt_item, (list, tuple)) and len(gt_item) == 0):
                continue

            gt = _first_item(gt_item)

            # Build input sequence tensor [1, T]
            seq = list(user_seq[-max_len:])
            seq = [0] * (max_len - len(seq)) + seq
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

            # Candidate set: GT + num_neg random negatives not in the user history
            rated = set(user_seq)
            rated.add(0)
            rated.add(gt)
            candidates = [gt]
            while len(candidates) < (num_neg + 1):
                x = random.randint(1, itemnum)
                if x not in rated:
                    candidates.append(x)
                    rated.add(x)

            cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

            # Score
            seq_output = model(seq_tensor)[:, -1, :]                 # [1, H]
            test_item_emb = item_emb(cand_tensor)                    # [C, H]
            scores = torch.matmul(seq_output, test_item_emb.T).squeeze(0).detach().cpu()

            order = torch.argsort(scores, descending=True).tolist()
            ranked_items = [candidates[i] for i in order]
            topk = ranked_items[:min(top_k, len(ranked_items))]

            if gt in topk:
                HR.append(1)
                REC.append(1)   # Recall@20 is identical to HR@20 in LOO eval
                idx = ranked_items.index(gt)
                NDCG.append(1.0 / np.log2(idx + 2))
            else:
                HR.append(0)
                REC.append(0)
                NDCG.append(0.0)

    return {
        f"HR@{top_k}": float(np.mean(HR)),
        f"NDCG@{top_k}": float(np.mean(NDCG)),
        f"Recall@{top_k}": float(np.mean(REC))
    }

def evaluate_er5(model, user_seqs: Iterable[Iterable[int]],
                 itemnum: int, device: torch.device, target_item: int,
                 top_k: int = 5, num_neg: int = 100):
    """
    Exposure Ratio@K for a fixed target item:
      - For each user sequence, rank the target vs. `num_neg` random negatives
      - Return fraction of users where target appears in Top-K
    """
    model.eval()
    exposed, total = 0, 0
    max_len = getattr(model, "max_seq_len", 50)
    item_emb = _get_item_embedding_layer(model)

    with torch.no_grad():
        for user_seq in user_seqs:
            if not user_seq:
                continue
            total += 1

            # Build input sequence tensor [1, T]
            seq = list(user_seq[-max_len:])
            seq = [0] * (max_len - len(seq)) + seq
            seq_tensor = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

            # Candidate set: target + negatives not in the user history
            rated = set(user_seq)
            rated.add(0)
            rated.add(target_item)
            candidates = [int(target_item)]
            while len(candidates) < (num_neg + 1):
                x = random.randint(1, itemnum)
                if x not in rated:
                    candidates.append(x)
                    rated.add(x)

            cand_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

            # Score
            seq_output = model(seq_tensor)[:, -1, :]                 # [1, H]
            test_item_emb = item_emb(cand_tensor)                    # [C, H]
            scores = torch.matmul(seq_output, test_item_emb.T).squeeze(0).detach().cpu()

            order = torch.argsort(scores, descending=True).tolist()
            ranked_items = [candidates[i] for i in order]
            if target_item in ranked_items[:min(top_k, len(ranked_items))]:
                exposed += 1

    return {"ER@5": float(exposed / max(1, total))}
