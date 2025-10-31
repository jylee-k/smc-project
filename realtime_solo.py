import os, json, time, numpy as np, yaml, librosa, torch
from typing import Optional, List, Dict
import sys
from panns_inference import SoundEventDetection
import torchaudio
from torch_vggish_yamnet import vggish
from torch_vggish_yamnet.input_proc import WaveformToInput
import csv
device = "cuda" if torch.cuda.is_available() else "cpu"

def to_pcm16(w):
    w = np.clip(np.asarray(w, np.float32), -1, 1)
    return (w * 32767).astype(np.int16).tobytes()

class LocalPANN:
    def __init__(self, device="cuda", min_seconds=1.0):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model = SoundEventDetection(checkpoint_path='./ast/pretrained_models/finetuned_panns.pth', device=self.device)
        self.min_seconds = float(min_seconds)
        try:
            self.labels = self.model.labels
        except Exception:
            self.labels = None

    def infer_clipwise(self, wave: np.ndarray, sr: int) -> np.ndarray:
        if sr != 16000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
            sr = 16000
        min_len = int(self.min_seconds * sr)
        if wave.shape[0] < min_len:
            pad = np.zeros((min_len - wave.shape[0],), np.float32)
            wave = np.concatenate([wave, pad], axis=0)
        with torch.no_grad():
            out = self.model.inference(torch.tensor(wave).unsqueeze(0))
        fw = out[0]
        probs = fw.max(axis=0)  # [1, 527]
        return probs

class LocalVGGish:
    def __init__(self, cfg, device="cuda"):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.cfg = cfg or {}

        labels_json = self.cfg.get('labels_json')
        label_csv = self.cfg.get('label_csv', 'ast/egs/audioset/class_labels_indices.csv')
        self.full_labels: Optional[List[str]] = None
        self._name_to_idx: Dict[str, int] = {}

        if label_csv and os.path.exists(label_csv):
            try:
                import csv
                with open(label_csv, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)[1:]
                self.full_labels = [row[2].strip().strip('"') for row in rows]
                self._name_to_idx = {name: i for i, name in enumerate(self.full_labels)}
            except Exception:
                self.full_labels = None
                self._name_to_idx = {}

        custom_labels: Optional[List[str]] = None
        if labels_json and os.path.exists(labels_json):
            try:
                with open(labels_json, 'r', encoding='utf-8') as f:
                    custom_labels = json.load(f)
            except Exception:
                custom_labels = None

        self.custom_indices: Optional[np.ndarray] = None
        if custom_labels and self._name_to_idx:
            keep = [self._name_to_idx[name] for name in custom_labels if name in self._name_to_idx]
            if len(keep) == len(custom_labels):
                self.custom_indices = np.asarray(keep, dtype=np.int64)
                self.labels = list(custom_labels)
            else:
                self.labels = self.full_labels or list(custom_labels)
        else:
            self.labels = self.full_labels

        self.model = vggish.get_vggish(with_classifier=True, pretrained=True).to(self.device)
        full_ckpt = self.cfg.get('full_ckpt') or './ast/pretrained_models/finetuned_vggish.pt'
        if not full_ckpt or not os.path.exists(full_ckpt):
            raise RuntimeError("custom_model.full_ckpt not found; please set cfg['custom_model']['full_ckpt']")
        ckpt = torch.load(full_ckpt, map_location=self.device)
        state = ckpt.get('model_state', ckpt)
        if isinstance(state, dict):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        try:
            self.model.load_state_dict(state, strict=False)
        except Exception:
            pass
        self.model.eval()

    @torch.no_grad()
    def infer_clipwise(self, wave: np.ndarray, sr: int) -> np.ndarray:
        # Build VGGish patches with WaveformToInput to exactly match training
        x = torch.tensor(wave, dtype=torch.float32).unsqueeze(0)  # [1, T]
        converter = WaveformToInput()
        patches = converter(x, sr)  # [N, 1, 96, 64]
        if patches.shape[0] == 0:
            return np.zeros((527,), dtype=np.float32)
        with torch.no_grad():
            logits = self.model(patches.to(self.device))
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = torch.sigmoid(logits)      # [N, 527]
            clip_probs = probs.max(dim=0).values    # [527]
        return clip_probs.detach().cpu().numpy()

class LocalAST:
    def __init__(self, device="cuda", mel_bins: int = 128, target_length: int = 1024,
                 checkpoint_path: str = "ast/pretrained_models/audio_mdl.pth",
                 label_csv: str = "ast/egs/audioset/class_labels_indices.csv"):
        # avoid clashing with Python stdlib module `ast` by importing via sys.path
        repo_root = os.path.dirname(os.path.abspath(__file__))
        ast_src = os.path.join(repo_root, "ast", "src")
        if ast_src not in sys.path:
            sys.path.append(ast_src)
        try:
            from models.ast_models import ASTModel  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to import AST model: {e}")

        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.mel_bins = int(mel_bins)
        self.target_length = int(target_length)

        # Load labels
        self.labels = None
        with open(label_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        # skip header
        self.labels = [row[2] for row in rows[1:]]


        # Build model and load checkpoint
        self.model = ASTModel(label_dim=527, input_tdim=self.target_length,
                              imagenet_pretrain=False, audioset_pretrain=False)
        # Wrap to match checkpoint topology if needed
        dp = torch.nn.DataParallel(self.model)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            dp.load_state_dict(ckpt, strict=False)
            self.model = dp
        else:
            # fallback to internal audioset-pretrained loading if available
            self.model = ASTModel(label_dim=527, input_tdim=self.target_length,
                                  imagenet_pretrain=True, audioset_pretrain=True)
        self.model = self.model.to(self.device).eval()

    def _fbank_from_wave(self, wave: np.ndarray, sr: int) -> torch.Tensor:
        if sr != 16000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
            sr = 16000
        waveform = torch.tensor(wave, dtype=torch.float32).unsqueeze(0)  # [1, T]
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=self.mel_bins, dither=0.0, frame_shift=10)

        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_length, :]

        # normalization from AST demo
        fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
        return fbank

    def infer_clipwise(self, wave: np.ndarray, sr: int) -> np.ndarray:
        feats = self._fbank_from_wave(wave, sr)
        x = feats.unsqueeze(0).to(self.device)  # [1, T, F]
        with torch.no_grad():
            out = self.model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            probs = torch.sigmoid(out).squeeze(0).detach().cpu().numpy()
        return probs

class RealTimeSolo:
    def __init__(self, cfg_path="configs/moe.yaml"):
        with open(cfg_path, "r",encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        st = self.cfg.get("stream", {})
        self.sr = int(st.get("sample_rate", 16000))
        self.chunk_ms = int(st.get("chunk_ms", 200))
        self.out_jsonl = st.get("out_jsonl", "runs/stream_preds.jsonl")
        self.out_json  = st.get("out_json",  "runs/stream_summary.json")
        os.makedirs(os.path.dirname(self.out_jsonl) or ".", exist_ok=True)

        self.local = LocalPANN(device=device, min_seconds=1.0)
        self.class_list = getattr(self.local, "labels", None)
        custom_cfg = dict(self.cfg.get('custom_model', {}) or {})
        tiers_path = custom_cfg.get('label_tiers', 'label_tiers.json')
        tier_labels: Optional[List[str]] = None
        if tiers_path and os.path.exists(tiers_path):
            try:
                with open(tiers_path, 'r', encoding='utf-8') as f:
                    tiers = json.load(f)
                tier1 = tiers.get('tier1') or []
                tier2 = tiers.get('tier2') or []
                tier_labels = list(tier1) + list(tier2)
            except Exception:
                tier_labels = None
        self.custom = LocalVGGish(device=device, cfg=custom_cfg)
        if tier_labels and getattr(self.custom, '_name_to_idx', None):
            keep = [self.custom._name_to_idx.get(lbl) for lbl in tier_labels if self.custom._name_to_idx.get(lbl) is not None]
            if len(keep) == len(tier_labels):
                self.custom.custom_indices = np.asarray(keep, dtype=np.int64)
                self.custom.labels = list(tier_labels)
        self.custom_labels = tier_labels if tier_labels else getattr(self.custom, 'labels', None)
        self.ast = LocalAST(device=device)
        self.ast_labels = self.ast.labels
        self.win_seconds = 1.0
        self.win_samples = int(self.sr * self.win_seconds)
        self.win_buffer = np.zeros((0,), np.float32)

        self.started = False
        self.session_t0 = None
        self.frame_idx = 0
        self.fout = None

        # Build mapping from base label spaces -> custom label space (if provided)
        self._pann_sel_idx = None
        self._ast_sel_idx = None
        self._vgg_sel_idx = None
        if self.custom_labels:
            try:
                self._pann_sel_idx = self._build_label_index(self.class_list, self.custom_labels)
            except Exception:
                self._pann_sel_idx = None
            try:
                self._ast_sel_idx = self._build_label_index(self.ast_labels, self.custom_labels)
            except Exception:
                self._ast_sel_idx = None
            try:
                base_vgg = getattr(self.custom, 'full_labels', None) or getattr(self.custom, 'labels', None)
                self._vgg_sel_idx = self._build_label_index(base_vgg, self.custom_labels)
            except Exception:
                self._vgg_sel_idx = None

        # Fusion weights (MoE-style static gating). Config example:
        # fusion:
        #   weights: { PANN: 1.0, VGGish: 1.0, AST: 1.0 }
        fusion_cfg = self.cfg.get('fusion', {}) if isinstance(self.cfg, dict) else {}
        w_cfg = fusion_cfg.get('weights', {}) if isinstance(fusion_cfg, dict) else {}
        self.fusion_weights = {
            'PANN': float(w_cfg.get('PANN', 1.0)),
            'VGGish': float(w_cfg.get('VGGish', 1.0)),
            'AST': float(w_cfg.get('AST', 1.0)),
        }

    def _normalize_label(self, s: str) -> str:
        return (s or "").strip().lower()

    def _build_label_index(self, base_labels, target_labels):
        if not base_labels or not target_labels:
            return None
        name_to_idx = {self._normalize_label(n): i for i, n in enumerate(base_labels)}
        result = []
        for n in target_labels:
            result.append(name_to_idx.get(self._normalize_label(n)))
        return result

    def _restrict_to_custom_softmax(self, full_probs: np.ndarray, sel_idx: list) -> np.ndarray:
        if sel_idx is None:
            return None
        out = np.zeros((len(sel_idx),), dtype=np.float32)
        for k, i in enumerate(sel_idx):
            if i is not None and 0 <= i < len(full_probs):
                out[k] = float(full_probs[i])
            else:
                out[k] = 0.0
        eps = 1e-6
        p = np.clip(out, eps, 1.0 - eps)
        logits = np.log(p) - np.log(1.0 - p)
        m = float(np.max(logits))
        ex = np.exp(logits - m)
        denom = float(np.sum(ex)) if float(np.sum(ex)) > 0 else 1.0
        return ex / denom

    def _sigmoid_to_softmax(self, probs: np.ndarray) -> np.ndarray:
        eps = 1e-6
        p = np.clip(np.asarray(probs, dtype=np.float32), eps, 1.0 - eps)
        logits = np.log(p) - np.log(1.0 - p)
        m = float(np.max(logits))
        ex = np.exp(logits - m)
        s = float(np.sum(ex))
        return ex / (s if s > 0 else 1.0)

    def _topk(self, dist: np.ndarray, labels: list, k: int = 3):
        if dist is None:
            return []
        k = int(min(k, len(dist)))
        idx = np.argsort(-dist)[:k]
        out = []
        for i in idx:
            lbl = labels[i] if labels and i < len(labels) else f'class_{int(i)}'
            out.append({
                'label': lbl,
                'score': float(dist[int(i)]),
                'index': int(i),
            })
        return out

    def start_session(self):
        if self.started: return
        self.fout = open(self.out_jsonl, "a", encoding="utf-8")
        self.session_t0 = time.time()
        self.frame_idx = 0
        self.started = True

    def stop_session(self):
        if not self.started: return
        try:
            self.fout.close()
        except:
            pass
        summary = {
            "start_ts": self.session_t0,
            "stop_ts": time.time(),
            "frames": self.frame_idx,
            "config": {"sr": self.sr, "chunk_ms": self.chunk_ms}
        }
        with open(self.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self.started = False

    def process_chunk(self, wave_chunk: np.ndarray, chunk_sr: int = None):
        assert self.started, "you have to first start_session()"
        if chunk_sr is not None and chunk_sr != self.sr:
            wave_chunk = librosa.resample(wave_chunk, orig_sr=chunk_sr, target_sr=self.sr)

        win_sec = self.chunk_ms / 1000.0
        t_start = self.frame_idx * win_sec
        t_end   = t_start + win_sec

        self.win_buffer = np.concatenate([self.win_buffer, wave_chunk], axis=0)
        if self.win_buffer.shape[0] > self.win_samples:
            self.win_buffer = self.win_buffer[-self.win_samples:]
        if self.win_buffer.shape[0] < self.win_samples:
            pad = np.zeros((self.win_samples - self.win_buffer.shape[0],), np.float32)
            local_window = np.concatenate([pad, self.win_buffer], axis=0)
        else:
            local_window = self.win_buffer

        local_prob = self.local.infer_clipwise(local_window, sr=self.sr)
        custom_prob = self.custom.infer_clipwise(local_window, sr=self.sr)
        ast_prob = self.ast.infer_clipwise(local_window, sr=self.sr)

        # Restrict PANN distribution to custom labels (if available)
        pann8 = self._restrict_to_custom_softmax(local_prob, self._pann_sel_idx) if self._pann_sel_idx is not None else None
        if pann8 is not None and self.custom_labels:
            pann_dist = pann8
            pann_labels = self.custom_labels
        else:
            pann_dist = self._sigmoid_to_softmax(local_prob)
            pann_labels = self.class_list
        local_top_idx = int(np.argmax(pann_dist))
        local_top_lbl = pann_labels[local_top_idx] if pann_labels else f"class_{local_top_idx}"
        local_top_score = float(pann_dist[local_top_idx])

        vgg8 = self._restrict_to_custom_softmax(custom_prob, self._vgg_sel_idx) if self._vgg_sel_idx is not None else None
        if vgg8 is not None and self.custom_labels:
            vgg_dist = vgg8
            custom_labels = self.custom_labels
        else:
            vgg_dist = self._sigmoid_to_softmax(custom_prob)
            custom_labels = getattr(self.custom, 'labels', None)
        custom_top_idx = int(np.argmax(vgg_dist))
        custom_top_lbl = custom_labels[custom_top_idx] if custom_labels else f"class_{custom_top_idx}"
        custom_top_score = float(vgg_dist[custom_top_idx])

        # Restrict AST distribution to custom labels (if available)
        ast8 = self._restrict_to_custom_softmax(ast_prob, self._ast_sel_idx) if self._ast_sel_idx is not None else None
        if ast8 is not None and self.custom_labels:
            ast_dist = ast8
            ast_labels = self.custom_labels
        else:
            ast_dist = self._sigmoid_to_softmax(ast_prob)
            ast_labels = self.ast_labels
        ast_top_idx = int(np.argmax(ast_dist))
        ast_top_lbl = ast_labels[ast_top_idx] if ast_labels else f"class_{ast_top_idx}"
        ast_top_score = float(ast_dist[ast_top_idx])

        fused_dist = None
        fused_labels = None
        if self.custom_labels:
            # prepare per-expert distributions in 8-class space when available
            dists = []
            weights = []
            # PANN
            if pann8 is not None:
                dists.append(pann8)
                weights.append(self.fusion_weights.get('PANN', 1.0))
            # VGGish (assume aligned to custom_labels size)
            if vgg_dist is not None and len(vgg_dist) == len(self.custom_labels):
                dists.append(vgg_dist)
                weights.append(self.fusion_weights.get('VGGish', 1.0))
            # AST
            if ast8 is not None:
                dists.append(ast8)
                weights.append(self.fusion_weights.get('AST', 1.0))
            if dists:
                W = np.asarray(weights, dtype=np.float32)
                W = W / (np.sum(W) if np.sum(W) > 0 else 1.0)
                stack = np.stack(dists, axis=0)
                fused_dist = (W[:, None] * stack).sum(axis=0)
                s = float(np.sum(fused_dist))
                fused_dist = fused_dist / (s if s > 0 else 1.0)
                fused_labels = self.custom_labels

        # Top3 lists
        pann_top3 = self._topk(pann_dist, pann_labels, k=3)
        vgg_top3 = self._topk(vgg_dist, custom_labels, k=3)
        ast_top3 = self._topk(ast_dist, ast_labels, k=3)
        fused_top3 = self._topk(fused_dist, fused_labels, k=3) if fused_dist is not None else []

        row = {
            "time_start": round(t_start, 3),
            "time_end": round(t_end, 3),
            "PANN": {
                "top_label": local_top_lbl,
                "top_score": round(local_top_score, 4),
                "top_index": local_top_idx,
                "top3": pann_top3
            },
            "VGGish": {
                "top_label": custom_top_lbl,
                "top_score": round(custom_top_score, 4),
                "top_index": custom_top_idx,
                "top3": vgg_top3
            },
            "AST": {
                "top_label": ast_top_lbl,
                "top_score": round(ast_top_score, 4),
                "top_index": ast_top_idx,
                "top3": ast_top3
            },
            **({
                "Fused": {
                    "top_label": (fused_labels[int(np.argmax(fused_dist))] if fused_labels else None),
                    "top_score": (float(fused_dist[int(np.argmax(fused_dist))]) if fused_dist is not None else None),
                    "top_index": (int(np.argmax(fused_dist)) if fused_dist is not None else None),
                    "top3": fused_top3
                }
            } if fused_dist is not None else {}),
        }
        self.fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.fout.flush()
        self.frame_idx += 1
        return row

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config.yaml")
    parser.add_argument("--wav", default=r".\raw_wav\Vehicle\1PQCymDynPs_240.00_250.00.wav")
    args = parser.parse_args()

    solo = RealTimeSolo(args.cfg)
    solo.start_session()
    try:
        y, sr = librosa.load(args.wav, sr=solo.sr, mono=True)
        hop = int(solo.sr * solo.chunk_ms / 1000.0)
        for off in range(0, len(y), hop):
            chunk = y[off: off + hop]
            if len(chunk) < hop:
                chunk = np.pad(chunk, (0, hop - len(chunk)))
            out = solo.process_chunk(chunk, chunk_sr=solo.sr)
        print("End, saved in /runs")
    finally:
        solo.stop_session()
