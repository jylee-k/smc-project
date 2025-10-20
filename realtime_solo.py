import os, json, time, numpy as np, yaml, librosa, torch
from panns_inference import SoundEventDetection
from torch_vggish_yamnet import yamnet

def to_pcm16(w):
    w = np.clip(np.asarray(w, np.float32), -1, 1)
    return (w * 32767).astype(np.int16).tobytes()

class LocalPANN:
    def __init__(self, device="cuda", min_seconds=1.0):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model = SoundEventDetection(checkpoint_path=None, device=self.device)
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
        if isinstance(out, dict) and "clipwise_output" in out:
            probs = out["clipwise_output"]
        elif isinstance(out, (list, tuple)) and "clipwise_output" in out[0]:
            probs = out[0]["clipwise_output"]
        else:
            tensor = out if isinstance(out, torch.Tensor) else out[0]
            tensor = torch.as_tensor(tensor)
            probs = torch.sigmoid(tensor)
        return probs.detach().cpu().numpy()[0]

class LocalYAMNet:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.model = yamnet.yamnet(pretrained=True).eval().to(self.device)
        self.labels = None
        yaml_path = './configs/yamnet_category_meta.yaml'
        with open(yaml_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        class_names = [item["name"] for item in meta]
        self.labels = class_names

    def infer_clipwise(self, wave: np.ndarray, sr: int) -> np.ndarray:
        if sr != 16000:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=16000)
        waveform = torch.tensor(wave, dtype=torch.float32).view(1, 1, -1, 1).to(self.device)  # shape: [1, 1, T, 1]

        with torch.no_grad():
            embedding, logits = self.model(waveform)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        return probs

class RealTimeSolo:
    def __init__(self, cfg_path="configs/config.yaml"):
        with open(cfg_path, "r",encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        st = self.cfg.get("stream", {})
        self.sr = int(st.get("sample_rate", 16000))
        self.chunk_ms = int(st.get("chunk_ms", 200))
        self.out_jsonl = st.get("out_jsonl", "runs/stream_preds.jsonl")
        self.out_json  = st.get("out_json",  "runs/stream_summary.json")
        os.makedirs(os.path.dirname(self.out_jsonl) or ".", exist_ok=True)
        self.device = 'cuda' if torch.cuda.is_available() else "cpu"

        self.local = LocalPANN(device=self.device, min_seconds=1.0)
        self.class_list = getattr(self.local, "labels", None)
        self.yamnet = LocalYAMNet(device=self.device)
        self.yam_labels = self.yamnet.labels
        self.win_seconds = 1.0
        self.win_samples = int(self.sr * self.win_seconds)
        self.win_buffer = np.zeros((0,), np.float32)

        self.started = False
        self.session_t0 = None
        self.frame_idx = 0
        self.fout = None

    def start_session(self):
        if self.started: 
            return
        self.fout = open(self.out_jsonl, "a", encoding="utf-8")
        self.session_t0 = time.time()
        self.frame_idx = 0
        self.win_buffer = np.zeros((0,), np.float32)  # reset buffer
        self.started = True

    def stop_session(self):
        if not self.started: 
            return
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
        yamnet_prob = self.yamnet.infer_clipwise(local_window, sr=self.sr)

        local_top_idx = int(np.argmax(local_prob))
        local_top_lbl = self.class_list[local_top_idx] if self.class_list else f"class_{local_top_idx}"
        local_top_score = float(local_prob[local_top_idx])

        yamnet_top_idx = int(np.argmax(yamnet_prob))
        yamnet_top_lbl = self.yam_labels[yamnet_top_idx]
        yamnet_top_score = float(yamnet_prob[yamnet_top_idx])

        row = {
            "time_start": round(t_start, 3),
            "time_end": round(t_end, 3),
            "PANN": {
                "top_label": local_top_lbl,
                "top_score": round(local_top_score, 4),
                "top_index": local_top_idx
            },
            "YAMNet": {
                "top_label": yamnet_top_lbl,
                "top_score": round(yamnet_top_score, 4),
                "top_index": yamnet_top_idx
            }
        }
        label = yamnet_top_lbl.lower()
        if ("alarm" in label) or ("siren" in label) or ("gunshot" in label):
            row["tier"] = 3
            row["type"] = "Critical"
            row["message"] = f"{yamnet_top_lbl} detected"
        elif ("baby" in label and "cry" in label):
            row["tier"] = 2
            row["type"] = "Warning"
            row["message"] = "Baby crying detected"
        else:
            row["tier"] = 1
            row["type"] = "Info"
            # If the label is something generic like "Speech" or "Noise"
            row["message"] = f"{yamnet_top_lbl} detected"
        
        
        
        self.fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.fout.flush()
        self.frame_idx += 1
        return row

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config.yaml")
    parser.add_argument("--wav", default=r"resources_R9_ZSCveAHg_7s.wav")
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
        print("模拟结束")
    finally:
        solo.stop_session()
