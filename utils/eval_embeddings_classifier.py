import os
import json
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class EmbeddingNpzDatasetEval(Dataset):
    def __init__(self, npz_dir: str, n_class: int = 527, pool: str = 'mean',
                 label_csv: Optional[str] = None,
                 keep_indices: Optional[List[int]] = None,
                 only_pos: bool = False):
        self.paths = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
        self.paths.sort()
        self.n_class = int(n_class)
        self.pool = pool
        self.mid_to_index: Optional[Dict[str, int]] = None
        self.keep_indices = sorted(list(set(keep_indices))) if keep_indices else None
        self.only_pos = bool(only_pos)
        if label_csv:
            # Optional: map mids to indices if labels are strings
            self.mid_to_index = {}
            with open(label_csv, 'r', encoding='utf-8') as f:
                next(f)
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        idx, mid, _ = parts[:3]
                        try:
                            self.mid_to_index[mid] = int(idx)
                        except Exception:
                            continue

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        data = np.load(p, allow_pickle=True)
        x = data['embedding']  # [T,128]
        y = data['labels']
        # pool embedding
        if self.pool == 'mean':
            feat = x.mean(axis=0)
        elif self.pool == 'max':
            feat = x.max(axis=0)
        else:
            feat = x.mean(axis=0)

        # full 527-d target first
        target_full = np.zeros((self.n_class,), dtype=np.float32)
        for lab in np.array(y).tolist():
            try:
                j = int(lab)
            except Exception:
                if self.mid_to_index is not None:
                    j = self.mid_to_index.get(str(lab), -1)
                else:
                    j = -1
            if 0 <= j < self.n_class:
                target_full[j] = 1.0

        # restrict to kept indices if provided
        if self.keep_indices:
            k = len(self.keep_indices)
            target = np.zeros((k,), dtype=np.float32)
            idx_map = {old: i for i, old in enumerate(self.keep_indices)}
            pos_any = False
            for old_idx in np.where(target_full > 0.5)[0].tolist():
                if old_idx in idx_map:
                    target[idx_map[old_idx]] = 1.0
                    pos_any = True
            if (not pos_any) and self.only_pos:
                return None
            return feat.astype(np.float32), target
        else:
            if (np.sum(target_full) <= 0.0) and self.only_pos:
                return None
            return feat.astype(np.float32), target_full


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int = 128, n_class: int = 527, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_class),
        )

    def forward(self, x):
        return self.net(x)


def collate_skip_none(batch):
    try:
        from torch.utils.data import default_collate
    except Exception:
        from torch.utils.data._utils.collate import default_collate  # type: ignore
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    total_top1_pos_correct = 0
    total_top1_pos_count = 0
    # micro accumulators
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0
    # macro accumulators per class
    first_batch = True
    for batch in loader:
        if batch is None:
            continue
        feat, tgt = batch
        feat = feat.to(device)
        tgt = tgt.to(device)
        logits = model(feat)
        probs = torch.sigmoid(logits)

        # top-1 on positives only
        # mask for samples having at least one positive
        pos_mask = (tgt.sum(dim=1) > 0)
        if pos_mask.any():
            probs_pos = probs[pos_mask]
            tgt_pos = tgt[pos_mask]
            top1_idx = probs_pos.argmax(dim=1)
            # correct if predicted class is one of the positives
            row_idx = torch.arange(top1_idx.shape[0], device=top1_idx.device)
            correct = tgt_pos[row_idx, top1_idx].sum().item()
            total_top1_pos_correct += int(correct)
            total_top1_pos_count += int(top1_idx.shape[0])

        # micro F1 at threshold
        pred = (probs >= threshold).to(tgt.dtype)
        micro_tp += int(((pred == 1) & (tgt == 1)).sum().item())
        micro_fp += int(((pred == 1) & (tgt == 0)).sum().item())
        micro_fn += int(((pred == 0) & (tgt == 1)).sum().item())

        # prepare for macro if needed
        if first_batch:
            n_class = tgt.shape[1]
            macro_tp = torch.zeros(n_class, dtype=torch.long, device=device)
            macro_fp = torch.zeros(n_class, dtype=torch.long, device=device)
            macro_fn = torch.zeros(n_class, dtype=torch.long, device=device)
            first_batch = False
        macro_pred = (probs >= threshold).to(torch.int64)
        macro_tgt = tgt.to(torch.int64)
        macro_tp += (macro_pred.eq(1) & macro_tgt.eq(1)).sum(dim=0)
        macro_fp += (macro_pred.eq(1) & macro_tgt.eq(0)).sum(dim=0)
        macro_fn += (macro_pred.eq(0) & macro_tgt.eq(1)).sum(dim=0)

    # compute metrics
    top1_pos_acc = (total_top1_pos_correct / total_top1_pos_count) if total_top1_pos_count > 0 else 0.0
    micro_precision = micro_tp / max(1, (micro_tp + micro_fp))
    micro_recall = micro_tp / max(1, (micro_tp + micro_fn))
    micro_f1 = (2 * micro_precision * micro_recall / max(1e-12, (micro_precision + micro_recall))) if (micro_precision + micro_recall) > 0 else 0.0

    if first_batch:
        macro_f1 = 0.0
    else:
        macro_precision = (macro_tp.float() / torch.clamp((macro_tp + macro_fp).float(), min=1.0)).cpu().numpy()
        macro_recall = (macro_tp.float() / torch.clamp((macro_tp + macro_fn).float(), min=1.0)).cpu().numpy()
        macro_f1_per_class = np.where(
            (macro_precision + macro_recall) > 0,
            2 * macro_precision * macro_recall / np.clip((macro_precision + macro_recall), 1e-12, None),
            0.0,
        )
        macro_f1 = float(np.mean(macro_f1_per_class))

    return {
        'top1_pos_acc': float(top1_pos_acc),
        'micro_f1@{:.2f}'.format(threshold): float(micro_f1),
        'macro_f1@{:.2f}'.format(threshold): float(macro_f1),
        'pos_samples': int(total_top1_pos_count),
    }


def resolve_keep_indices(args) -> Optional[List[int]]:
    keep_indices: Optional[List[int]] = None
    if args.keep_indices:
        keep_indices = [int(s) for s in args.keep_indices.split(',') if s.strip()]
    elif args.keep_names and args.label_csv:
        name_to_idx: Dict[str, int] = {}
        with open(args.label_csv, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    idx, _mid, disp = parts[:3]
                    disp = disp.strip().strip('"')
                    try:
                        name_to_idx[disp] = int(idx)
                    except Exception:
                        continue
        keep_names_str = args.keep_names or ""
        raw_names = keep_names_str.split(",")
        names = [n.strip() for n in raw_names]
        names = [n for n in names if n]
        keep_indices = []
        for n in names:
            if n in name_to_idx:
                keep_indices.append(name_to_idx[n])
        if not keep_indices:
            raise SystemExit('No keep_names matched label_csv display_name entries')
    return keep_indices


def main():
    ap = argparse.ArgumentParser(description='Evaluate trained VGGish MLP classifier on NPZ embeddings')
    ap.add_argument('--npz_dir', default=r"D:\NUS_1\CS5647_Sound_and_Music\smc-project\data\features\audioset_v1_embeddings\bal_train\npz", help='Directory of validation/test .npz files')
    ap.add_argument('--ckpt', default=r'runs/emb_clf/ckpt_epoch_88.pt', help='Path to checkpoint .pt saved by training script')
    ap.add_argument('--n_class', type=int, default=527)
    ap.add_argument('--pool', default='mean', choices=['mean', 'max'])
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=1)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--label_csv', default=r"D:\NUS_1\CS5647_Sound_and_Music\smc-project\ast\egs\audioset\data\class_labels_indices.csv")
    ap.add_argument('--keep_indices', default=None)
    ap.add_argument('--keep_names', default='Alarm, Fire alarm, Doorbell, Knock, Baby cry, Telephone bell ringing, Vehicle horn, Civil defense siren')
    ap.add_argument('--only_pos', default=True, help='Keep only samples with at least one positive among kept classes')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    keep_indices = resolve_keep_indices(args)
    effective_n_class = len(keep_indices) if keep_indices else args.n_class

    ds = EmbeddingNpzDatasetEval(
        args.npz_dir,
        n_class=args.n_class,
        pool=args.pool,
        label_csv=args.label_csv,
        keep_indices=keep_indices,
        only_pos=args.only_pos,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_skip_none,
    )

    # Model
    model = MLPClassifier(in_dim=128, n_class=effective_n_class, hidden=args.hidden, dropout=args.dropout).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    sd = ckpt.get('model_state', ckpt)
    model.load_state_dict(sd, strict=False)

    metrics = evaluate(model, loader, device, threshold=float(args.threshold))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

