import os
import json
import time
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class EmbeddingNpzDataset(Dataset):
    def __init__(self, npz_dir: str, n_class: int = 527, pool: str = 'mean',
                 label_csv: Optional[str] = None,
                 keep_indices: Optional[List[int]] = None, drop_no_pos: bool = False):
        self.paths = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
        self.paths.sort()
        self.n_class = int(n_class)
        self.pool = pool
        self.mid_to_index: Optional[Dict[str, int]] = None
        self.keep_indices = sorted(list(set(keep_indices))) if keep_indices else None
        self.drop_no_pos = drop_no_pos
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
        x = data['embedding']  # [T,128] float32 in [0,1]
        y = data['labels']     # list/array of ints or mids
        # pool embedding
        if self.pool == 'mean':
            feat = x.mean(axis=0)
        elif self.pool == 'max':
            feat = x.max(axis=0)
        else:
            # default mean
            feat = x.mean(axis=0)

        # full 527-d target first
        target_full = np.zeros((self.n_class,), dtype=np.float32)
        # labels could be ints or strings
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

        # If keeping subset of classes, remap to compact range
        if self.keep_indices:
            k = len(self.keep_indices)
            target = np.zeros((k,), dtype=np.float32)
            idx_map = {old: i for i, old in enumerate(self.keep_indices)}
            pos_any = False
            for old_idx in np.where(target_full > 0.5)[0].tolist():
                if old_idx in idx_map:
                    target[idx_map[old_idx]] = 1.0
                    pos_any = True
            if (not pos_any) and self.drop_no_pos:
                # return None to signal skipping this sample
                return None
            return feat.astype(np.float32), target
        else:
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
    """Filter out None items; return None if all filtered."""
    try:
        from torch.utils.data import default_collate
    except Exception:
        from torch.utils.data._utils.collate import default_collate  # type: ignore
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)


def train_one_epoch(model: nn.Module, loader: DataLoader, device: str, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        if batch is None:
            continue
        feat, tgt = batch
        feat = feat.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(feat)
        loss = criterion(logits, tgt)
        loss.backward()
        optimizer.step()
        bs = feat.shape[0]
        total_loss += loss.item() * bs
        total_n += bs
    print(total_n)
    return {"loss": total_loss / max(1, total_n)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        if batch is None:
            continue
        feat, tgt = batch
        feat = feat.to(device)
        tgt = tgt.to(device)
        logits = model(feat)
        loss = criterion(logits, tgt)
        bs = feat.shape[0]
        total_loss += loss.item() * bs
        total_n += bs
    return {"val_loss": total_loss / max(1, total_n)}


def main():
    ap = argparse.ArgumentParser(description="Train multilabel classifier on AudioSet v1 VGGish embeddings (.npz)")
    ap.add_argument('--train_npz_dir', default=r"D:\NUS_1\CS5647_Sound_and_Music\smc-project\data\features\audioset_v1_embeddings\bal_train\npz")
    ap.add_argument('--val_npz_dir', default=None)
    ap.add_argument('--n_class', type=int, default=527)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--num_workers', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--pool', default='mean', choices=['mean', 'max'])
    ap.add_argument('--label_csv', default=r"D:\NUS_1\CS5647_Sound_and_Music\smc-project\ast\egs\audioset\data\class_labels_indices.csv", help='Optional class_labels_indices.csv to map mids and names')
    ap.add_argument('--keep_indices', default=None, help='Comma-separated class indices to keep (e.g., 0,137,321)')
    ap.add_argument('--keep_names', default='Alarm, Fire alarm, Doorbell, Knock, Baby cry, Telephone bell ringing, Vehicle horn, Civil defense siren', help='Comma-separated display names to keep (requires --label_csv)')
    ap.add_argument('--drop_no_pos', default='True', help='Drop samples without any kept positive labels')
    ap.add_argument('--save_dir', default='runs/emb_clf')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Resolve keep_indices from names or explicit indices
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
                    # CSV display_name may have quotes
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

    # If keeping subset, override n_class to subset size
    effective_n_class = len(keep_indices) if keep_indices else args.n_class

    # parse drop_no_pos to bool
    drop_no_pos = str(args.drop_no_pos).lower() in ("1", "true", "yes", "y")

    train_ds = EmbeddingNpzDataset(
        args.train_npz_dir, n_class=args.n_class, pool=args.pool,
        label_csv=args.label_csv,
        keep_indices=keep_indices, drop_no_pos=drop_no_pos,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_skip_none,
    )

    model = MLPClassifier(in_dim=128, n_class=effective_n_class, hidden=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: List[Dict] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_log = train_one_epoch(model, train_loader, device, optimizer)
        log = {"epoch": epoch, **train_log, "seconds": round(time.time() - t0, 2)}
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'history': history,
        }, os.path.join(args.save_dir, f'ckpt_epoch_{epoch}.pt'))

    with open(os.path.join(args.save_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
