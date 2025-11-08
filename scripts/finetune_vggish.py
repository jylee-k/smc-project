import os
import json
import csv
import time
import argparse
from typing import List, Dict, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


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


class RawWaveDataset(Dataset):
    def __init__(self, data_root: str, labels: List[str], name_to_full_index: Dict[str, int]):
        import torchaudio
        self.data_root = data_root
        self.labels = labels
        self.name_to_full_index = name_to_full_index

        items: List[Tuple[str, int]] = []
        for name in labels:
            cand = os.path.join(data_root, _slugify(name))
            if os.path.isdir(cand):
                paths = list_audio_files(cand)
            else:
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
        import torchaudio
        path, y = self.items[idx]
        try:
            wav, sr = torchaudio.load(path)  # [C, T], native sr
            return wav, int(sr), int(y)
        except Exception:
            try:
                import librosa
                x, sr = librosa.load(path, sr=None, mono=False)
                import numpy as np
                if isinstance(x, np.ndarray):
                    if x.ndim == 1:
                        wav = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                    else:
                        wav = torch.tensor(x, dtype=torch.float32)
                else:
                    return None
                return wav, int(sr), int(y)
            except Exception:
                return None


def collate_waves(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    waves = [b[0] for b in batch]
    srs = [int(b[1]) for b in batch]
    ys = torch.tensor([int(b[2]) for b in batch], dtype=torch.long)
    return waves, srs, ys


@torch.no_grad()
def evaluate(model, waves: List[torch.Tensor], srs: List[int], ys: torch.Tensor, device: str) -> Tuple[float, float]:
    model.eval()
    from torch_vggish_yamnet.input_proc import WaveformToInput
    conv = WaveformToInput()
    total = 0
    correct = 0
    total_loss = 0.0
    crit = nn.CrossEntropyLoss(reduction='sum')
    for i in range(len(waves)):
        try:
            patches = conv(waves[i].float(), srs[i])  # [n_i, 1, 96, 64]
        except Exception:
            continue
        if patches.shape[0] == 0:
            continue
        logits_p = model(patches.to(device))
        if isinstance(logits_p, (list, tuple)):
            logits_p = logits_p[0]
        probs = torch.sigmoid(logits_p)
        p = probs.mean(dim=0)
        p = p.clamp(1e-8, 1 - 1e-8)
        clip_logits = torch.log(p / (1 - p))
        y = ys[i].to(device)
        loss = crit(clip_logits.unsqueeze(0), y.unsqueeze(0))
        total_loss += float(loss.item())
        pred = int(clip_logits.argmax().item())
        correct += int(pred == int(ys[i]))
        total += 1
    return total_loss / max(1, total), correct / max(1, total)


def train_one_epoch(model, loader, device, optimizer):
    model.train()
    from torch_vggish_yamnet.input_proc import WaveformToInput
    conv = WaveformToInput()
    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0
    correct = 0
    for batch in loader:
        if batch is None:
            continue
        waves, srs, ys = batch
        patch_list: List[torch.Tensor] = []
        counts: List[int] = []
        kept_y: List[int] = []
        for i in range(len(waves)):
            try:
                p = conv(waves[i].float(), int(srs[i]))  # [n_i, 1, 96, 64]
            except Exception:
                continue
            if p.shape[0] == 0:
                continue
            counts.append(int(p.shape[0]))
            kept_y.append(int(ys[i]))
            patch_list.append(p)
        if not patch_list:
            continue
        patches = torch.cat(patch_list, dim=0)

        optimizer.zero_grad(set_to_none=True)
        logits_p = model(patches.to(device))
        if isinstance(logits_p, (list, tuple)):
            logits_p = logits_p[0]
        start = 0
        clip_logits = []
        for i, n_i in enumerate(counts):
            seg = logits_p[start:start + n_i]
            start += n_i
            logit = seg.max(dim=0).values
            clip_logits.append(logit.unsqueeze(0))
        clip_logits = torch.cat(clip_logits, dim=0)  # [B, 527]

        y = torch.tensor(kept_y, dtype=torch.long, device=device)
        loss = crit(clip_logits, y)
        loss.backward()
        optimizer.step()

        bs = y.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs
        pred = clip_logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
    print(total_n)
    return {"loss": total_loss / max(1, total_n), "acc": correct / max(1, total_n)}


def main():
    ap = argparse.ArgumentParser(description='Fine-tune VGGish classifier')
    ap.add_argument('--labels_json', default='data/custom_label.json')
    ap.add_argument('--data_root', default='dataset')
    ap.add_argument('--label_csv', default='ast/egs/audioset/class_labels_indices.csv', help='AudioSet class_labels_indices.csv for name→index mapping')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--save_dir', default='runs/vggish_cls')
    ap.add_argument('--unfreeze', choices=['all', 'head_only'], default='all')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open(args.labels_json, 'r', encoding='utf-8') as f:
        labels: List[str] = json.load(f)

    name_to_full_index: Dict[str, int] = {}
    with open(args.label_csv, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)[1:]
    for row in rows:
        try:
            idx = int(row[0])
            disp = row[2].strip().strip('"')
            name_to_full_index[disp] = idx
        except Exception:
            continue
    missing = [n for n in labels if n not in name_to_full_index]
    if missing:
        raise SystemExit(f"Labels not found in label_csv: {missing}")

    ds = RawWaveDataset(args.data_root, labels, name_to_full_index)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_waves)

    from torch_vggish_yamnet import vggish
    model = vggish.get_vggish(with_classifier=True, pretrained=True).to(device)

    if args.unfreeze == 'head_only':
        for n, p in model.named_parameters():
            p.requires_grad = ('classifier' in n)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history: List[Dict] = []
    best = {"train_acc": -1.0, "epoch": 0}
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, loader, device, optimizer)
        log = {"epoch": epoch, **tr, "seconds": round(time.time() - t0, 2)}
        history.append(log)
        print(json.dumps(log, ensure_ascii=False))

        if tr['acc'] > best['train_acc']:
            best = {"train_acc": tr['acc'], "epoch": epoch}
            state = model.state_dict()
            torch.save({"model": state, "model_state": state}, os.path.join(args.save_dir, 'best.pt'))

    with open(os.path.join(args.save_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(json.dumps({"best": best}, ensure_ascii=False))


if __name__ == '__main__':
    main()
