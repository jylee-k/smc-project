import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np
import librosa

sr = 16000  # Sample rate
seconds = 5  # Duration
channels = 1 # Mono

# -------------------------
# 1) Record or load audio
# -------------------------
def record_audio(seconds, sr=16000, channels=1):
    """
    Records from the default microphone.
    Returns: float32 numpy array shape (samples,) at given sr, mono/stereo as requested.
    """
    print(f"Recording {seconds}s @ {sr} Hz, channels={channels} ...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=channels, dtype="float32")
    sd.wait()
    audio = np.squeeze(audio)  # (samples, ) if mono
    return audio, sr

def load_audio(path):
    """
    Loads an audio file with original sampling rate (no resampling).
    Returns: audio (float32), sr
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    audio = np.squeeze(audio)
    return audio, sr

def save_audio(audio, sr):
    write("output.wav", sr, audio)  # Save as WAV file
    print("Saved as output.wav")
    
# -----------------------------------------
# 2) Convert to mono + resample to 16 kHz
# -----------------------------------------
def to_mono_16k(audio, sr, target_sr=16000):
    """
    Downmix to mono and resample to target_sr (default 16k).
    """
    # Downmix to mono if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # Ensure float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio, sr


'''audio, sr = record_audio(seconds, sr, channels)
audio2raw, sr2raw = load_audio("demo.wav")
audio2, sr2 = to_mono_16k(audio2raw, sr2raw)
print(type(audio))
sd.wait()
save_audio(audio2, sr2)'''


# --------------------------------------------
# Real-time recorder that emits fixed chunks
# --------------------------------------------
import threading, queue, time

class ChunkRecorder:
    """
    Record from mic in real-time and emit fixed-size chunks.

    Usage:
        rec = ChunkRecorder(sr=16000, channels=1, chunk_size_sec=2.0)
        rec.start()       # begin recording
        ... (recording) ...
        rec.stop()        # stop and flush remainder
        chunks = rec.chunks  # list of np.ndarrays (float32, mono)
    """
    def __init__(self, sr=16000, channels=1, chunk_size_sec=2.0, blocksize=0, on_chunk=None):
        self.sr = sr
        self.channels = channels
        self.chunk_size_sec = float(chunk_size_sec)
        self.chunk_samples = int(round(self.sr * self.chunk_size_sec))
        if self.chunk_samples <= 0:
            raise ValueError("chunk_size_sec must be > 0")

        self._q = queue.Queue()
        self._stop_evt = threading.Event()
        self._worker = None
        self._stream = None

        self._buf = np.empty((0,), dtype=np.float32)  # mono buffer
        self.chunks = []  # public: filled during/after recording
        self.blocksize = blocksize  # 0 lets PortAudio choose
        self.on_chunk = on_chunk

    def _callback(self, indata, frames, time_info, status):
        if status:
            # non-fatal info (XRuns, etc.)
            print(f"[audio status] {status}")
        # ensure mono float32 shape (samples,)
        x = indata.astype(np.float32)
        if x.ndim == 2 and x.shape[1] > 1:
            x = np.mean(x, axis=1)
        else:
            x = np.squeeze(x)
        self._q.put(x)

    def _worker_loop(self):
        while not self._stop_evt.is_set():
            try:
                block = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            # append & emit fixed chunks
            self._buf = np.concatenate([self._buf, block])
            while len(self._buf) >= self.chunk_samples:
                chunk = self._buf[:self.chunk_samples].copy()
                self.chunks.append(chunk)
                self._buf = self._buf[self.chunk_samples:]
                if self.on_chunk:
                    self.on_chunk(chunk)  # infer

        # flush remainder (if any) as final shorter chunk
        if len(self._buf) > 0:
            self.chunks.append(self._buf.copy())
        self._buf = np.empty((0,), dtype=np.float32)

    def start(self):
        if self._stream is not None:
            return  # already running
        self.chunks = []
        self._stop_evt.clear()

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

        self._stream = sd.InputStream(
            samplerate=self.sr,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
        )
        self._stream.start()
        print(f"[ChunkRecorder] Started @ {self.sr} Hz, chunk={self.chunk_size_sec}s")

    def stop(self):
        if self._stream is None:
            return
        self._stop_evt.set()
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
        # wait worker to flush remainder
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        print(f"[ChunkRecorder] Stopped. Total chunks: {len(self.chunks)}")
        
# -------------------------------
# Rec button Tkinter interface
# -------------------------------
import tkinter as tk
from tkinter import messagebox
from realtime_solo import RealTimeSolo
class RecordButton:
    def __init__(self, master, sr=16000, channels=1, chunk_size_sec=2.0):
        self.master = master
        self.master.title("Recorder")
        self.sr = sr
        self.channels = channels
        self.chunk_size_sec = float(chunk_size_sec)

        self.recorder = None
        self.is_recording = False

        self.btn = tk.Button(master, text="Start", width=18, command=self.toggle)
        self.btn.pack(padx=20, pady=20)

        self.info = tk.Label(master, text=f"SR: {sr} Hz  |  Chunk: {chunk_size_sec}s")
        self.info.pack(pady=(0, 10))

        self.save_var = tk.IntVar(value=0)
        self.chk = tk.Checkbutton(master, text="Save chunks as WAV on stop", variable=self.save_var)
        self.chk.pack()

        self.solo = RealTimeSolo("config.yaml")
        self.solo.start_session()

    def toggle(self):
        if not self.is_recording:
            # Start
            try:
                self.recorder = ChunkRecorder(sr=self.sr, channels=self.channels, chunk_size_sec=self.chunk_size_sec, on_chunk=self.handle_chunk)
                self.recorder.start()
                self.is_recording = True
                self.btn.config(text="Stop")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            # Stop
            try:
                if self.recorder:
                    self.recorder.stop()
            finally:
                self.is_recording = False
                self.btn.config(text="Start")
                chunks = self.recorder.chunks if self.recorder else []
                total_sec = sum(len(c) for c in chunks) / float(self.sr) if chunks else 0.0
                print(f"[UI] Got {len(chunks)} chunks, total {total_sec:.2f}s")

                if self.save_var.get() and chunks:
                    # Save each chunk as output_chunkXX.wav using your current writer style
                    for i, c in enumerate(chunks):
                        fname = f"output_chunk{i+1:02d}.wav"
                        write(fname, self.sr, c)  # uses scipy.io.wavfile.write you already imported
                        print(f"Saved {fname} ({len(c)/self.sr:.2f}s)")

                messagebox.showinfo("Done", f"Recorded {len(chunks)} chunks.\nCheck console for details.")

    def handle_chunk(self, chunk):
        self.solo.process_chunk(chunk, chunk_sr=16000)

def run_record_button_ui(sr=16000, channels=1, chunk_size_sec=2.0):
    root = tk.Tk()
    app = RecordButton(root, sr=sr, channels=channels, chunk_size_sec=chunk_size_sec)
    root.mainloop()
    
if __name__ == "__main__":
    # Launch minimal GUI (Start/Stop). Optional: check "Save chunks as WAV on stop".
    run_record_button_ui(sr=16000, channels=1, chunk_size_sec=2.0)


