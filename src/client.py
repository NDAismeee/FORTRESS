import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
from model.sasrec import SASRec


class RecDataset(Dataset):
    def __init__(self, user_seq, itemnum, maxlen):
        self.sequences, self.positives, self.negatives = [], [], []
        self.maxlen, self.itemnum = maxlen, itemnum

        for seq in user_seq:
            if not seq:
                continue
            # Logic: N items sinh ra N-1 samples
            for i in range(1, len(seq)):
                subseq = seq[:i][-maxlen:]
                pad_len = maxlen - len(subseq)
                padded_seq = [0] * pad_len + subseq
                pos_item = seq[i]
                neg_item = np.random.randint(1, itemnum + 1)
                while neg_item in seq:
                    neg_item = np.random.randint(1, itemnum + 1)
                self.sequences.append(padded_seq)
                self.positives.append(pos_item)
                self.negatives.append(neg_item)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.positives[idx], dtype=torch.long),
            torch.tensor(self.negatives[idx], dtype=torch.long),
        )


class Client:
    def __init__(self, client_id, local_data, model_config, train_config, device, attack=None):
        """
        attack: None, or an attack object implementing:
            - augment_sequences(local_sequences, model) -> List[List[int]]
            - poison_loss(h_last, model) -> scalar tensor
            - cfg.beta (float)
        """
        self.id = client_id
        self.device = device
        self.model = SASRec(**model_config).to(device)
        self.train_config = train_config
        self.local_data = local_data
        self.attack = attack  # attack strategy (PSMU / A-hum / combined)

    def set_weights(self, state_dict):
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def get_weights(self):
        # Fix OOM: Trả về CPU tensor để giải phóng GPU ngay lập tức
        return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    # ===== helpers =====
    def _infonce_inbatch(self, z1, z2, tau: float):
        """
        SimCLR-style InfoNCE with in-batch negatives.
        z1, z2: [B, H], already L2-normalized.
        """
        B = z1.size(0)
        if B < 2:
            return z1.new_zeros([])

        reps = torch.cat([z1, z2], dim=0)                 # [2B, H]
        logits = (reps @ reps.t()) / max(tau, 1e-6)       # [2B, 2B]
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        mask = torch.eye(2*B, device=z1.device, dtype=torch.bool)
        logits = logits.masked_fill(mask, -1e9)

        labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z1.device)
        return F.cross_entropy(logits, labels)

    def _seq_augment(self, seq, p_mask=0.15, min_crop=0.6, p_reorder=0.15):
        """
        Simple sequence augment: crop (keep tail), random mask, local swap.
        """
        B, T = seq.size()
        aug = seq.clone()
        lengths = (aug != 0).sum(dim=1)
        for i in range(B):
            L = int(lengths[i].item())
            if L <= 1:
                continue
            # crop tail
            keep_L = max(1, int(round(L * min_crop)))
            drop = L - keep_L
            if drop > 0:
                idxs = (aug[i] != 0).nonzero(as_tuple=False).squeeze(-1)
                aug[i, idxs[:drop]] = 0
            # mask
            if p_mask > 0:
                mask_flag = (torch.rand(T, device=seq.device) < p_mask) & (aug[i] != 0)
                aug[i][mask_flag] = 0
            # local reorder (single swap)
            if p_reorder > 0 and L > 2 and np.random.rand() < p_reorder:
                idxs = (aug[i] != 0).nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
                if idxs.size > 1:
                    a, b = np.random.choice(idxs, 2, replace=False)
                    tmp = aug[i, a].item()
                    aug[i, a] = aug[i, b]
                    aug[i, b] = tmp
        return aug

    def train(self):
        # (optional) attack aug: append synthetic sequences
        local_sequences = list(self.local_data)
        if self.attack:
            try:
                local_sequences = self.attack.augment_sequences(local_sequences, self.model)
            except Exception as e:
                print(f"[Client {self.id}] Attack augment failed: {e}")

        dataset = RecDataset(
            local_sequences,
            self.train_config['num_items'],
            self.train_config['max_seq_len']
        )

        # === SỬA 1: Hạ ngưỡng lọc để tận dụng dữ liệu ===
        # Vì dataset tạo N-1 mẫu từ N items. Nếu min_len=5 (ở train.py) -> tạo ra 4 mẫu.
        # Đặt min_samples=4 để không bỏ sót các user nhỏ này.
        min_samples = 4  
        
        if len(dataset) < min_samples:
            # print(f"Client {self.id} skipped (len={len(dataset)})")
            # === SỬA 2: Trả về None thay vì 0.0 để Server lọc bỏ ===
            return self.get_weights(), None

        loader = DataLoader(
            dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=True,
            drop_last=False,
            pin_memory=True
        )

        # ---- hyper ----
        lr         = float(self.train_config.get('lr', 1e-3))
        tau        = float(self.train_config.get('temperature', 0.2))
        lam_tcr    = float(self.train_config.get('lambda_tcr', 0.5))
        lam_seq    = float(self.train_config.get('lambda_seq_view', 0.1))
        lam_usr    = float(self.train_config.get('lambda_user_view', 0.1))
        lam_itm    = float(self.train_config.get('lambda_item_view', 0.2))
        eps_fgsm   = float(self.train_config.get('item_fgsm_eps', 0.1))
        clip_norm  = float(self.train_config.get('grad_clip', 5.0))

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()

        self.model.train()
        
        # === SỬA 3: Đưa biến total_loss ra ngoài vòng lặp epoch ===
        total_loss = 0.0
        num_epochs = self.train_config['local_epochs']
        # Tính tổng số bước thực tế sẽ chạy
        total_steps = len(loader) * num_epochs

        for epoch in range(num_epochs):
            for batch_seq, batch_pos, batch_neg in loader:
                batch_seq = batch_seq.to(self.device)
                batch_pos = batch_pos.to(self.device)
                batch_neg = batch_neg.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                # forward
                seq_output = self.model(batch_seq)                        # [B,T,H]
                B, T, H = seq_output.size()
                lengths = (batch_seq != 0).sum(dim=1)
                last_idx = (lengths - 1).clamp(min=0)
                rows = torch.arange(B, device=self.device)
                h_last = seq_output[rows, last_idx]                       # [B,H]

                # ===== Base next-item BCE =====
                pos_e = self.model.item_embedding(batch_pos)              # [B,H]
                neg_e = self.model.item_embedding(batch_neg)              # [B,H]
                pos_logits = self.model.predict(seq_output, pos_e)        # [B]
                neg_logits = self.model.predict(seq_output, neg_e)        # [B]
                logits = torch.cat([pos_logits, neg_logits], dim=0)       # [2B]
                labels = torch.cat([
                    torch.ones_like(pos_logits),
                    torch.zeros_like(neg_logits)
                ], dim=0).float()
                loss = bce(logits, labels)

                # ===== Sequence view CL =====
                if lam_seq != 0.0:
                    seq1 = self._seq_augment(batch_seq)
                    seq2 = self._seq_augment(batch_seq)
                    o1 = self.model(seq1)
                    o2 = self.model(seq2)
                    l1 = (seq1 != 0).sum(dim=1).clamp(min=1) - 1
                    l2 = (seq2 != 0).sum(dim=1).clamp(min=1) - 1
                    h1 = o1[rows, l1]
                    h2 = o2[rows, l2]
                    z1 = self.model.project(h1)   # normalized
                    z2 = self.model.project(h2)
                    loss = loss + lam_seq * self._infonce_inbatch(z1, z2, tau)

                # ===== User view CL =====
                if lam_usr != 0.0:
                    noise_scale = 0.1
                    u1 = h_last + torch.randn_like(h_last) * noise_scale
                    u2 = h_last + torch.randn_like(h_last) * noise_scale
                    z1_u = self.model.project(u1)
                    z2_u = self.model.project(u2)
                    loss = loss + lam_usr * self._infonce_inbatch(z1_u, z2_u, tau)

                # ===== Item view CL (FGSM on pos item embedding) =====
                if lam_itm != 0.0:
                    pos_e_adv = pos_e.detach().clone().requires_grad_(True)
                    with torch.enable_grad():
                        pos_logits_adv = self.model.predict(seq_output.detach(), pos_e_adv)
                        loss_sur = -torch.mean(torch.log(torch.sigmoid(pos_logits_adv) + 1e-12))
                        grad, = torch.autograd.grad(loss_sur, pos_e_adv, retain_graph=False, create_graph=False)

                    v1 = (pos_e + eps_fgsm * torch.sign(grad)).detach()
                    v2 = (pos_e - eps_fgsm * torch.sign(grad)).detach()
                    z1_i = self.model.project(v1)
                    z2_i = self.model.project(v2)
                    loss = loss + lam_itm * self._infonce_inbatch(z1_i, z2_i, tau)

                # ===== Temporal Consistency Regularization =====
                if lam_tcr > 0:
                    valid = lengths >= 2
                    if valid.any():
                        prev_seq = batch_seq.clone()
                        prev_seq[rows, (lengths - 1).clamp_min(0)] = 0
                        seq_output_prev = self.model(prev_seq)
                        h_t = seq_output[rows[valid], (lengths - 1)[valid]]
                        h_prev = seq_output_prev[rows[valid], (lengths - 2)[valid]]
                        tcr_loss = F.mse_loss(h_t, h_prev, reduction="mean")
                        loss = loss + lam_tcr * tcr_loss

                # (optional) poison loss
                if self.attack:
                    try:
                        poison = self.attack.poison_loss(h_last, self.model)
                        beta = float(getattr(self.attack, "cfg", getattr(self.attack, "config", None)).beta) \
                               if hasattr(self.attack, "cfg") or hasattr(self.attack, "config") else 0.1
                        loss = loss + beta * poison
                    except Exception as e:
                        print(f"[Client {self.id}] Attack poison failed: {e}")

                # backward
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                if clip_norm and clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_norm)
                optimizer.step()

                total_loss += float(loss.detach().item())

        # === SỬA 4: Chia cho tổng số bước để lấy loss trung bình chính xác ===
        avg_loss = total_loss / max(1, total_steps)
        
        return self.get_weights(), avg_loss