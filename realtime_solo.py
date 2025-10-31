import os
import json
import time
import sys
import csv
import yaml
from typing import Optional, List, Dict

import numpy as np
import librosa
import torch
import torchaudio

# --- Local Model Imports ---
from panns_inference import SoundEventDetection
from torch_vggish_yamnet import vggish
from torch_vggish_yamnet.input_proc import WaveformToInput

device = "cuda" if torch.cuda.is_available() else "cpu"

def to_pcm16(w):
    """Utility function to convert float32 audio to int16 bytes."""
    w = np.clip(np.asarray(w, np.float32), -1, 1)
    return (w * 32767).astype(np.int16).tobytes()

class LocalPANN:
    """Wrapper for the PANNs SoundEventDetection model."""
    def __init__(self, device="cuda", min_seconds=1.0):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model = SoundEventDetection(checkpoint_path='./ast/pretrained_models/finetuned_panns.pth', device=self.device)
        self.min_seconds = float(min_seconds)
        try:
            self.labels = self.model.labels
        except Exception:
            self.labels = None

    def infer_clipwise(self, wave: np.ndarray, sr: int) -> np.ndarray:
        """
        Runs inference on a single audio clip.

        Args:
            wave (np.ndarray): Input audio waveform.
            sr (int): Sample rate of the waveform.

        Returns:
            np.ndarray: Clip-wise probabilities (sigmoid outputs).
        """
        if sr != 16000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        # Pad if shorter than minimum required length
        min_len = int(self.min_seconds * sr)
        if wave.shape[0] < min_len:
            pad = np.zeros((min_len - wave.shape[0],), np.float32)
            wave = np.concatenate([wave, pad], axis=0)
        
        with torch.no_grad():
            # Model expects [1, T] tensor
            out = self.model.inference(torch.tensor(wave).unsqueeze(0))
        
        fw = out[0]
        probs = fw.max(axis=0)  # Max-pool over time -> [1, 527]
        return probs

class LocalVGGish:
    """Wrapper for a fine-tuned VGGish classifier."""
    def __init__(self, cfg, device="cuda"):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.cfg = cfg or {}

        labels_json = self.cfg.get('labels_json')
        label_csv = self.cfg.get('label_csv', 'ast/egs/audioset/class_labels_indices.csv')
        self.full_labels: Optional[List[str]] = None
        self._name_to_idx: Dict[str, int] = {}

        # Load the full label set (e.g., AudioSet 527 classes)
        if label_csv and os.path.exists(label_csv):
            try:
                with open(label_csv, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)[1:] # skip header
                self.full_labels = [row[2].strip().strip('"') for row in rows]
                self._name_to_idx = {name: i for i, name in enumerate(self.full_labels)}
            except Exception:
                self.full_labels = None
                self._name_to_idx = {}

        # Load a custom subset of labels, if provided
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
                self.labels = list(custom_labels) # Use custom labels
            else:
                self.labels = self.full_labels or list(custom_labels) # Fallback
        else:
            self.labels = self.full_labels

        # Load VGGish model
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
            pass # Ignore errors if classifier head differs
        self.model.eval()

    @torch.no_grad()
    def infer_clipwise(self, wave: np.ndarray, sr: int) -> np.ndarray:
        """
        Runs inference on a single audio clip using VGGish features.

        Args:
            wave (np.ndarray): Input audio waveform.
            sr (int): Sample rate of the waveform.

        Returns:
            np.ndarray: Clip-wise probabilities (sigmoid outputs).
        """
        # Build VGGish patches with WaveformToInput to exactly match training
        x = torch.tensor(wave, dtype=torch.float32).unsqueeze(0)  # [1, T]
        converter = WaveformToInput()
        patches = converter(x, sr)  # [N, 1, 96, 64]
        
        if patches.shape[0] == 0:
            return np.zeros((527,), dtype=np.float32) # Return zeros if audio is too short
        
        with torch.no_grad():
            logits = self.model(patches.to(self.device))
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = torch.sigmoid(logits)      # [N, 527]
            clip_probs = probs.max(dim=0).values    # Max-pool over patches -> [527]
        return clip_probs.detach().cpu().numpy()

class LocalAST:
    """Wrapper for the AST (Audio Spectrogram Transformer) model."""
    def __init__(self, device="cuda", mel_bins: int = 128, target_length: int = 1024,
                 checkpoint_path: str = "ast/pretrained_models/audio_mdl.pth",
                 label_csv: str = "ast/egs/audioset/class_labels_indices.csv"):
        
        # This is a clever hack to avoid clashing with the Python stdlib `ast` module
        repo_root = os.path.dirname(os.path.abspath(__file__))
        ast_src = os.path.join(repo_root, "ast", "src")
        if ast_src not in sys.path:
            sys.path.append(ast_src)
        try:
            from models.ast_models import ASTModel  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to import AST model from {ast_src}: {e}")

        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.mel_bins = int(mel_bins)
        self.target_length = int(target_length) # Number of frames

        # Load labels
        self.labels = None
        with open(label_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        self.labels = [row[2] for row in rows[1:]] # skip header

        # Build model and load checkpoint
        self.model = ASTModel(label_dim=527, input_tdim=self.target_length,
                              imagenet_pretrain=False, audioset_pretrain=False)
        
        # Wrap in DataParallel to match checkpoint topology if it was saved from DP
        dp = torch.nn.DataParallel(self.model)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            dp.load_state_dict(ckpt, strict=False)
            self.model = dp
        else:
            # Fallback to loading the audioset-pretrained weights from the AST repo
            self.model = ASTModel(label_dim=527, input_tdim=self.target_length,
                                  imagenet_pretrain=True, audioset_pretrain=True)
        self.model = self.model.to(self.device).eval()

    def _fbank_from_wave(self, wave: np.ndarray, sr: int) -> torch.Tensor:
        """Converts a waveform to a filterbank tensor matching AST input."""
        if sr != 16000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        waveform = torch.tensor(wave, dtype=torch.float32).unsqueeze(0)  # [1, T]
        
        # Equivalent to VGGish/AudioSet feature extraction
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type='hanning', num_mel_bins=self.mel_bins, dither=0.0, frame_shift=10)

        # Pad or truncate to target_length
        n_frames = fbank.shape[0]
        p = self.target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p)) # Pad time dimension
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:self.target_length, :] # Truncate
        
        # Normalization from AST demo
        fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
        return fbank

    def infer_clipwise(self, wave: np.ndarray, sr: int) -> np.ndarray:
        """
        Runs inference on a single audio clip.

        Args:
            wave (np.ndarray): Input audio waveform.
            sr (int): Sample rate of the waveform.

        Returns:
            np.ndarray: Clip-wise probabilities (sigmoid outputs).
        """
        feats = self._fbank_from_wave(wave, sr)
        x = feats.unsqueeze(0).to(self.device)  # [1, T, F]
        with torch.no_grad():
            out = self.model(x)
            if isinstance(out, (list, tuple)):
                out = out[0]
            probs = torch.sigmoid(out).squeeze(0).detach().cpu().numpy() # [527]
        return probs

class RealTimeSolo:
    """
    Main pipeline class.
    Orchestrates multiple models (PANN, VGGish, AST), fuses their predictions,
    and writes results to a log file.
    """
    def __init__(self, cfg_path="configs/moe.yaml"):
        with open(cfg_path, "r",encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        
        st = self.cfg.get("stream", {})
        self.sr = int(st.get("sample_rate", 16000))
        self.chunk_ms = int(st.get("chunk_ms", 200)) # Note: app.py sends 2000ms
        self.out_jsonl = st.get("out_jsonl", "runs/stream_preds.jsonl")
        self.out_json  = st.get("out_json",  "runs/stream_summary.json")
        os.makedirs(os.path.dirname(self.out_jsonl) or ".", exist_ok=True)

        # --- 1. Initialize Models ---
        self.local = LocalPANN(device=device, min_seconds=1.0)
        self.class_list = getattr(self.local, "labels", None)
        
        custom_cfg = dict(self.cfg.get('custom_model', {}) or {})
        self.custom = LocalVGGish(device=device, cfg=custom_cfg)
        
        self.ast = LocalAST(device=device)
        self.ast_labels = self.ast.labels

        # --- 2. Load Custom Label Tiers (for fusion) ---
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
        
        # Use custom tier labels if available, otherwise fallback to VGGish labels
        self.custom_labels = tier_labels if tier_labels else getattr(self.custom, 'labels', None)
        
        # --- 3. Build Mappings to Custom Label Space ---
        # This creates indices to map 527-class outputs to the smaller custom_labels set
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

        # --- 4. Load Fusion Weights ---
        fusion_cfg = self.cfg.get('fusion', {}) if isinstance(self.cfg, dict) else {}
        w_cfg = fusion_cfg.get('weights', {}) if isinstance(fusion_cfg, dict) else {}
        self.fusion_weights = {
            'PANN': float(w_cfg.get('PANN', 1.0)),
            'VGGish': float(w_cfg.get('VGGish', 1.0)),
            'AST': float(w_cfg.get('AST', 1.0)),
        }

        # --- 5. Session State ---
        self.started = False
        self.session_t0 = None
        self.frame_idx = 0
        self.fout = None

    def _normalize_label(self, s: str) -> str:
        """Helper to lowercase and strip label names for matching."""
        return (s or "").strip().lower()

    def _build_label_index(self, base_labels, target_labels):
        """Creates a mapping from a base label list to a target list."""
        if not base_labels or not target_labels:
            return None
        name_to_idx = {self._normalize_label(n): i for i, n in enumerate(base_labels)}
        result = []
        for n in target_labels:
            result.append(name_to_idx.get(self._normalize_label(n))) # Appends index or None
        return result

    def _restrict_to_custom_softmax(self, full_probs: np.ndarray, sel_idx: list) -> np.ndarray:
        """
        Selects probabilities using the sel_idx mapping and applies softmax.
        This converts a 527-class sigmoid output to a e.g. 8-class softmax output.
        """
        if sel_idx is None:
            return None
        
        # Select the probabilities corresponding to the custom labels
        out = np.zeros((len(sel_idx),), dtype=np.float32)
        for k, i in enumerate(sel_idx):
            if i is not None and 0 <= i < len(full_probs):
                out[k] = float(full_probs[i])
            else:
                out[k] = 0.0 # Label not found in base model
        
        # Convert selected probabilities (sigmoids) to softmax
        eps = 1e-6
        p = np.clip(out, eps, 1.0 - eps)
        logits = np.log(p) - np.log(1.0 - p) # Logit approximation
        m = float(np.max(logits))
        ex = np.exp(logits - m)
        denom = float(np.sum(ex)) if float(np.sum(ex)) > 0 else 1.0
        return ex / denom

    def _sigmoid_to_softmax(self, probs: np.ndarray) -> np.ndarray:
        """Converts a full sigmoid probability array to softmax."""
        eps = 1e-6
        p = np.clip(np.asarray(probs, dtype=np.float32), eps, 1.0 - eps)
        logits = np.log(p) - np.log(1.0 - p) # Logit approximation
        m = float(np.max(logits))
        ex = np.exp(logits - m)
        s = float(np.sum(ex))
        return ex / (s if s > 0 else 1.0)

    def _topk(self, dist: np.ndarray, labels: list, k: int = 3):
        """Gets the top-k predictions from a probability distribution."""
        if dist is None:
            return []
        k = int(min(k, len(dist)))
        idx = np.argsort(-dist)[:k] # Get indices of top k scores
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
        """Opens the log file and resets session state."""
        if self.started: return
        self.fout = open(self.out_jsonl, "a", encoding="utf-8")
        self.session_t0 = time.time()
        self.frame_idx = 0
        self.started = True

    def stop_session(self):
        """Closes the log file and writes a session summary."""
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
        """
        The main processing function.
        Takes one chunk of audio, runs all models, fuses results, and writes to log.

        Args:
            wave_chunk (np.ndarray): The input audio chunk.
            chunk_sr (int, optional): The sample rate of the chunk.
        
        Returns:
            dict: The dictionary of results that was written to the log.
        """
        assert self.started, "you have to first start_session()"
        if chunk_sr is not None and chunk_sr != self.sr:
            wave_chunk = librosa.resample(wave_chunk, orig_sr=chunk_sr, target_sr=self.sr)

        win_sec = self.chunk_ms / 1000.0
        t_start = self.frame_idx * win_sec
        t_end   = t_start + win_sec

        # The Streamlit app sends a full 2.0s chunk, so we process it directly
        # instead of using the old internal buffering logic.
        local_window = wave_chunk

        # --- 1. Run Inference on all models ---
        local_prob = self.local.infer_clipwise(local_window, sr=self.sr)
        custom_prob = self.custom.infer_clipwise(local_window, sr=self.sr)
        ast_prob = self.ast.infer_clipwise(local_window, sr=self.sr)

        # --- 2. Process PANN Results ---
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

        # --- 3. Process VGGish Results ---
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

        # --- 4. Process AST Results ---
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

        # --- 5. Fuse Results (if custom labels are available) ---
        fused_dist = None
        fused_labels = None
        if self.custom_labels:
            # Prepare per-expert distributions in the custom label space
            dists = []
            weights = []
            
            if pann8 is not None:
                dists.append(pann8)
                weights.append(self.fusion_weights.get('PANN', 1.0))
            
            # VGGish distribution should already be aligned to custom_labels
            if vgg_dist is not None and len(vgg_dist) == len(self.custom_labels):
                dists.append(vgg_dist)
                weights.append(self.fusion_weights.get('VGGish', 1.0))

            if ast8 is not None:
                dists.append(ast8)
                weights.append(self.fusion_weights.get('AST', 1.0))
            
            if dists:
                # Calculate weighted average of distributions
                W = np.asarray(weights, dtype=np.float32)
                W = W / (np.sum(W) if np.sum(W) > 0 else 1.0) # Normalize weights
                stack = np.stack(dists, axis=0)
                fused_dist = (W[:, None] * stack).sum(axis=0)
                
                # Re-normalize final distribution
                s = float(np.sum(fused_dist))
                fused_dist = fused_dist / (s if s > 0 else 1.0)
                fused_labels = self.custom_labels

        # --- 6. Compile Final Results ---
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
            # Add Fused results only if fusion was successful
            **({
                "Fused": {
                    "top_label": (fused_labels[int(np.argmax(fused_dist))] if fused_labels and fused_dist is not None else None),
                    "top_score": (float(fused_dist[int(np.argmax(fused_dist))]) if fused_dist is not None else None),
                    "top_index": (int(np.argmax(fused_dist)) if fused_dist is not None else None),
                    "top3": fused_top3
                }
            } if fused_dist is not None else {}),
        }
        
        # Write to log file and flush
        self.fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.fout.flush()
        self.frame_idx += 1
        return row

if __name__ == "__main__":
    # --- Test script for processing a single WAV file ---
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--wav", default=r".\raw_wav\Vehicle\1PQCymDynPs_240.00_250.00.wav", help="Path to test WAV file")
    args = parser.parse_args()

    print(f"Loading pipeline with config: {args.cfg}")
    solo = RealTimeSolo(args.cfg)
    solo.start_session()
    print(f"Processing test file: {args.wav}")
    try:
        y, sr = librosa.load(args.wav, sr=solo.sr, mono=True)
        
        # Note: This logic processes 2.0s chunks, matching app.py
        chunk_samples = int(solo.sr * 2.0)
        
        if chunk_samples == 0:
            raise ValueError("Chunk size is zero, check config.")

        for off in range(0, len(y), chunk_samples):
            chunk = y[off: off + chunk_samples]
            if len(chunk) < chunk_samples:
                # Pad the final chunk
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            
            out = solo.process_chunk(chunk, chunk_sr=solo.sr)
            if out.get("Fused"):
                print(f"Time {out['time_start']:.2f}s: {out['Fused']['top_label']} ({out['Fused']['top_score']:.2f})")
            
        print(f"Processing complete. Results saved in {solo.out_jsonl}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        solo.stop_session()
        print("Session stopped.")