import os
import json
import argparse
import time
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from scripts.finetune_vggish import _slugify, list_audio_files


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _logits_from_sigmoid_probs(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log(1.0 - p)


def _derive_label_from_file(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return _slugify(stem)


def _load_user_audio(user_dir: str) -> Tuple[str, List[str]]:
    paths = list_audio_files(user_dir)
    if not paths:
        raise SystemExit(f"No audio found in '{user_dir}'. Please place a file there.")
    first = paths[0]
    label = _derive_label_from_file(first)
    return label, [first]


def _expand_and_finetune_vggish(user_label: str, audio_paths: List[str], save_dir: str,
                                epochs: int = 5, lr: float = 5e-4) -> None:
    from torch_vggish_yamnet import vggish
    import torchaudio
    from scripts.finetune_vggish import collate_waves as vgg_collate, train_one_epoch as vgg_train_one_epoch

    device = _device()
    _ensure_dir(save_dir)

    model = vggish.get_vggish(with_classifier=True, pretrained=True).to(device)

    ckpt_candidates = [
        'pretrained_model/finetuned_vggish.pt',
    ]
    state = None
    for p in ckpt_candidates:
        if os.path.exists(p):
            sd = torch.load(p, map_location=device)
            state = sd.get('model_state', sd.get('model', sd))
            break
    if state is not None:
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            pass

    last_linear_name = None
    last_linear = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_linear_name = name
            last_linear = m
    if last_linear is None:
        raise RuntimeError('VGGish last linear layer not found')

    in_f = int(last_linear.in_features)
    out_f_old = int(last_linear.out_features)
    out_f_new = out_f_old + 1
    new_linear = nn.Linear(in_f, out_f_new)
    with torch.no_grad():
        new_linear.weight[:out_f_old].copy_(last_linear.weight)
        new_linear.bias[:out_f_old].copy_(last_linear.bias)
        nn.init.normal_(new_linear.weight[out_f_old:], mean=0.0, std=0.01)
        nn.init.zeros_(new_linear.bias[out_f_old:])

    def _replace_module(root: nn.Module, dotted: str, new_mod: nn.Module):
        parts = dotted.split('.')
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_mod)

    _replace_module(model, last_linear_name, new_linear)

    for p in model.parameters():
        p.requires_grad = False
    for p in new_linear.parameters():
        p.requires_grad = True

    class UploadedWaveDataset(Dataset):
        def __init__(self, paths: List[str], label_index: int):
            self.paths = list(paths)
            self.y = int(label_index)
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx: int):
            import torchaudio
            path = self.paths[idx]
            try:
                wav, sr = torchaudio.load(path)
                return wav, int(sr), self.y
            except Exception:
                try:
                    import librosa
                    x, sr = librosa.load(path, sr=None, mono=False)
                    if isinstance(x, np.ndarray):
                        if x.ndim == 1:
                            wav = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                        else:
                            wav = torch.tensor(x, dtype=torch.float32)
                        return wav, int(sr), self.y
                except Exception:
                    return None

    optimizer = torch.optim.AdamW(new_linear.parameters(), lr=lr)
    model.train()
    new_idx = out_f_old  # the added class index
    history = []
    ds = UploadedWaveDataset(audio_paths, new_idx)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, collate_fn=vgg_collate)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = vgg_train_one_epoch(model, loader, device, optimizer)
        history.append({"epoch": ep, **tr, "seconds": round(time.time() - t0, 2)})

    # Save expanded model
    sd = model.state_dict()
    torch.save({"model": sd, "model_state": sd, "num_classes": out_f_new, "new_label": user_label, "history": history},
               os.path.join(save_dir, 'vggish_expanded.pt'))
    with open(os.path.join(save_dir, 'vggish_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({"num_classes": out_f_new, "new_index": new_idx, "new_label": user_label}, f, ensure_ascii=False, indent=2)


def _build_panns(classes_num: int, device: str):
    from panns_inference.models import Cnn14_DecisionLevelMax
    return Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, hop_size=320,
                                  mel_bins=64, fmin=50, fmax=14000, classes_num=classes_num).to(device)


def _expand_and_finetune_panns(user_label: str, audio_paths: List[str], save_dir: str,
                               epochs: int = 10, lr: float = 1e-4) -> None:
    """Expand PANNs head by 1 unit and fine-tune the head only."""
    import librosa
    from scripts.finetune_panns import train_one_epoch as panns_train_one_epoch
    device = _device()
    _ensure_dir(save_dir)

    # Load a 527-class model to grab head weights
    base_model = _build_panns(527, device)
    ckpt_candidates = [
        'pretrained_model/finetuned_panns.pth',
    ]
    loaded = False
    for p in ckpt_candidates:
        if os.path.exists(p):
            sd = torch.load(p, map_location=device)
            if isinstance(sd, dict) and 'model' in sd:
                sd = sd['model']
            try:
                base_model.load_state_dict(sd, strict=False)
                loaded = True
                break
            except Exception:
                pass

    # Build new model with +1 class and transfer head weights
    model = _build_panns(528, device)
    if loaded:
        state = base_model.state_dict()
        model_sd = model.state_dict()
        head_names = [
            'fc_audioset.weight', 'fc_audioset.bias',
            'clipwise_fc.weight', 'clipwise_fc.bias',
            'fc.weight', 'fc.bias'
        ]
        for k, v in state.items():
            if k in model_sd and model_sd[k].shape == v.shape and (k not in head_names):
                model_sd[k].copy_(v)
        for w_key, b_key in [('fc_audioset.weight', 'fc_audioset.bias'),
                             ('clipwise_fc.weight', 'clipwise_fc.bias'),
                             ('fc.weight', 'fc.bias')]:
            if w_key in model_sd and w_key in state and b_key in model_sd and b_key in state:
                W_old = state[w_key]
                b_old = state[b_key]
                W_new = model_sd[w_key]
                b_new = model_sd[b_key]
                with torch.no_grad():
                    W_new[:W_old.shape[0]].copy_(W_old)
                    b_new[:b_old.shape[0]].copy_(b_old)
                    nn.init.normal_(W_new[W_old.shape[0]:], mean=0.0, std=0.01)
                    nn.init.zeros_(b_new[b_old.shape[0]:])
        model.load_state_dict(model_sd, strict=False)

    for p in model.parameters():
        p.requires_grad = False
    head = None
    for name in ['fc_audioset', 'clipwise_fc', 'fc']:
        if hasattr(model, name) and isinstance(getattr(model, name), nn.Linear):
            head = getattr(model, name)
            break
    if head is None:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                head = m
    if head is None:
        raise RuntimeError('PANNs head not found')
    for p in head.parameters():
        p.requires_grad = True

    X_list: List[torch.Tensor] = []
    for p in audio_paths:
        x, sr = librosa.load(p, sr=32000, mono=True)
        samples = int(round(10.0 * 32000))
        if len(x) < samples:
            x = np.pad(x, (0, samples - len(x)))
        elif len(x) > samples:
            start = (len(x) - samples) // 2
            x = x[start:start + samples]
        X_list.append(torch.tensor(x, dtype=torch.float32))
    if not X_list:
        raise SystemExit('Failed to load audio for PANNs fine-tuning.')
    X = torch.stack(X_list, dim=0).to(device)  # [B, T]

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    model.train()
    new_idx = 527
    history = []
    y = torch.full((X.shape[0],), fill_value=new_idx, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(X.cpu(), y)
    loader = DataLoader(ds, batch_size=min(16, X.shape[0]), shuffle=True, num_workers=0, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr = panns_train_one_epoch(model, loader, device, optimizer, criterion)
        history.append({"epoch": ep, **tr, "seconds": round(time.time() - t0, 2)})

    sd = model.state_dict()
    torch.save({"model": sd, "num_classes": 528, "new_label": user_label, "history": history},
               os.path.join(save_dir, 'panns_expanded.pth'))
    with open(os.path.join(save_dir, 'panns_meta.json'), 'w', encoding='utf-8') as f:
        json.dump({"num_classes": 528, "new_index": new_idx, "new_label": user_label}, f, ensure_ascii=False, indent=2)


# ---------------- AST ----------------
def _extract_ast_fbank(wave: np.ndarray, sr: int, target_len: int = 1024, mel_bins: int = 128) -> torch.Tensor:
    import librosa
    import torch
    if sr != 16000:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
        sr = 16000
    waveform = torch.tensor(wave, dtype=torch.float32).unsqueeze(0)  # [1, T]
    import torchaudio
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
        window_type='hanning', num_mel_bins=mel_bins, dither=0.0, frame_shift=10)
    n_frames = fbank.shape[0]
    p = target_len - n_frames
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_len, :]
    fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
    return fbank


def main() -> None:
    parser = argparse.ArgumentParser(
        description='fine-tune on a user audio clip.'
    )
    parser.add_argument('--user_uploads', default='./user_uploads')
    parser.add_argument('--epochs_vggish', type=int, default=5)
    parser.add_argument('--epochs_panns', type=int, default=10)
    parser.add_argument('--lr_vggish', type=float, default=5e-4)
    parser.add_argument('--lr_panns', type=float, default=1e-4)
    parser.add_argument('--out_dir', default='pretrained_model/expanded_models')
    parser.add_argument('--models', nargs='+', default=['vggish', 'panns'], choices=['vggish', 'panns'])
    args = parser.parse_args()

    label, audio_paths = _load_user_audio(args.user_uploads)
    print(f"[Info] New label = {label}")
    print(f"[Info] Using file = {audio_paths[0]}")

    _ensure_dir(args.out_dir)

    if 'vggish' in args.models:
        print('[VGGish] Expanding last layer and fine-tuning...')
        _expand_and_finetune_vggish(label, audio_paths,
                                    save_dir=os.path.join(args.out_dir, 'vggish'),
                                    epochs=args.epochs_vggish, lr=args.lr_vggish)
        print('[VGGish] Done.')

    if 'panns' in args.models:
        print('[PANNs] Expanding head and fine-tuning...')
        _expand_and_finetune_panns(label, audio_paths,
                                   save_dir=os.path.join(args.out_dir, 'panns'),
                                   epochs=args.epochs_panns, lr=args.lr_panns)
        print('[PANNs] Done.')
    with open(os.path.join(args.out_dir, 'new_label_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({"label": label, "sources": audio_paths, "time": time.time()}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
