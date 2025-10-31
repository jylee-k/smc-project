import os
import json
import time
import random
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
def _slugify(name: str) -> str:
    s = name.strip().replace('/', '_')
    s = ''.join(ch if (ch.isalnum() or ch == '_') else '_' for ch in s)
    while '__' in s:
        s = s.replace('__', '_')
    return s.strip('_')


def list_audio_files(root: str) -> List[str]:
    out: List[str] = []
    for r, _d, fns in os.walk(root):
        for f in fns:
            if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                out.append(os.path.join(r, f))
    out.sort()
    return out


class WaveDataset(Dataset):
    def __init__(self, data_root: str, labels: List[str], name_to_full_index: Dict[str, int], sr: int = 32000, seconds: float = 10.0):
        import librosa
        self.data_root = data_root
        self.labels = labels
        # map display name -> original 0..526 AudioSet index
        self.name_to_full_index = name_to_full_index
        self.sr = int(sr)
        self.samples = int(round(seconds * self.sr))
        self.librosa = librosa

        items: List[Tuple[str, int]] = []
        for name in labels:
            # prefer underscore folder name
            cand = os.path.join(data_root, _slugify(name))
            if os.path.isdir(cand):
                paths = list_audio_files(cand)
            else:
                # try simple space->underscore
                alt = os.path.join(data_root, name.replace(',', '').replace('  ', ' ').replace(' ', '_'))
                paths = list_audio_files(alt) if os.path.isdir(alt) else []
            for p in paths:
                full_idx = int(self.name_to_full_index[name])
                items.append((p, full_idx))

        if not items:
            raise RuntimeError(f"No audio found under '{data_root}' for labels: {labels}")
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        wav, sr = self.librosa.load(path, sr=self.sr, mono=True)
        # center-crop or pad to fixed length
        n = wav.shape[0]
        if n < self.samples:
            pad = np.zeros((self.samples - n,), dtype=np.float32)
            x = np.concatenate([wav, pad], axis=0)
        elif n > self.samples:
            start = (n - self.samples) // 2
            x = wav[start:start + self.samples]
        else:
            x = wav
        x = torch.tensor(x, dtype=torch.float32)
        return x, int(y)


def build_panns_model(checkpoint: str, device: str = 'cpu') -> nn.Module:
    # Try to import Cnn14 from panns_inference
    try:
        from panns_inference.models import Cnn14_DecisionLevelMax
        model = Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, hop_size=320,
                      mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    except Exception as e:
        raise RuntimeError(f"Failed to import PANNS Cnn14: {e}")

    sd = torch.load(checkpoint, map_location=device)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    model.load_state_dict(sd, strict=False)

    # Keep original 527-dim head; fine-tune with subset targets mapped to 0..526
    return model


def split_indices(n: int, val_ratio: float, seed: int = 42):
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_val = int(round(n * val_ratio))
    return idxs[n_val:], idxs[:n_val]


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_n = 0
    correct = 0
    for x, y_idx in loader:
        x = x.to(device)
        y_idx = y_idx.to(device)
        B = x.size(0)
        optimizer.zero_grad(set_to_none=True)
        y = torch.zeros(B, 527, device=device, dtype=torch.float32)
        y.scatter_(1, y_idx.unsqueeze(1), 1.0)  # (B, C)
        # Cnn14 in panns expects [B, data_length] waveform
        out = model(x, mixup_lambda = None)
        logits = out['clipwise_output']
        bce_elem = F.binary_cross_entropy(logits, y, reduction='none')
        loss = bce_elem.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * B
        total_n += B
        pred = logits.argmax(dim=1)
        correct += (pred == y_idx).sum().item()
    return {"loss": total_loss / max(1, total_n), "acc": correct / max(1, total_n)}



def main():
    ap = argparse.ArgumentParser(description='Fine-tune PANNS Cnn14 on custom labels (single-label)')
    ap.add_argument('--labels_json', default='data/custom_label.json')
    ap.add_argument('--data_root', default='raw_wav')
    ap.add_argument('--checkpoint', default='ast/pretrained_models/Cnn14_DecisionLevelMax.pth')
    ap.add_argument('--label_csv', default='ast/egs/audioset/class_labels_indices.csv', help='AudioSet class_labels_indices.csv for name→index mapping')
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save_dir', default='runs/panns_finetune')
    ap.add_argument('--unfreeze', choices=['all', 'head'], default='all')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.labels_json, 'r', encoding='utf-8') as f:
        labels: List[str] = json.load(f)
    # map display name -> full AudioSet index (0..526)
    name_to_full_index: Dict[str, int] = {}
    import csv
    with open(args.label_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)[1:]
    for row in rows:
        try:
            idx = int(row[0])
            disp = row[2].strip().strip('"')
            name_to_full_index[disp] = idx
        except Exception:
            continue
    # ensure all custom labels exist in mapping
    missing = [n for n in labels if n not in name_to_full_index]
    if missing:
        raise SystemExit(f"Labels not found in label_csv: {missing}")

    full_ds = WaveDataset(args.data_root, labels, name_to_full_index, sr=32000, seconds=args.seconds)
    n = len(full_ds)
    train_idx = list(range(n))
    train_ds = torch.utils.data.Subset(full_ds, train_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    model = build_panns_model(checkpoint=args.checkpoint, device=device).to(device)
    if args.unfreeze == 'head':
        for p in model.parameters():
            p.requires_grad = False
        # unfreeze last linear
        head = None
        for name in ['fc_audioset', 'clipwise_fc', 'fc']:
            if hasattr(model, name) and isinstance(getattr(model, name), nn.Linear):
                head = getattr(model, name)
                break
        if head is None:
            # fallback: last linear
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    head = m
        if head is not None:
            for p in head.parameters():
                p.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best = {"train_acc": -1.0, "epoch": 0}
    history: List[Dict] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, device, optimizer, criterion)
        log = {"epoch": epoch, **tr, "seconds": round(time.time() - t0, 2)}
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))
        if tr["acc"] > best["train_acc"]:
            best = {"train_acc": tr["acc"], "epoch": epoch}
            sd =  model.state_dict()
            deploy_sd = {"model": sd}
            torch.save(deploy_sd, os.path.join(args.save_dir, "best.pth"))

    with open(os.path.join(args.save_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
