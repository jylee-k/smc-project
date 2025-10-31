import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import numpy as np
import librosa
import threading
import queue
import time

# --- Constants ---
sr = 16000  # Default sample rate
seconds = 5  # Default duration for simple recording
channels = 1  # Default to mono

# -------------------------
# 1) Simple Audio I/O Functions
# -------------------------

def record_audio(seconds, sr=16000, channels=1):
    """
    Records from the default microphone for a fixed duration.

    Args:
        seconds (int): Duration of the recording.
        sr (int): Sample rate.
        channels (int): Number of audio channels.

    Returns:
        tuple: (numpy.ndarray (float32), int)
               The recorded audio as a float32 array and the sample rate.
    """
    print(f"Recording {seconds}s @ {sr} Hz, channels={channels} ...")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=channels, dtype="float32")
    sd.wait()  # Wait for recording to finish
    audio = np.squeeze(audio)  # (samples, ) if mono
    return audio, sr

def load_audio(path):
    """
    Loads an audio file with its original sampling rate.

    Args:
        path (str): Path to the audio file.

    Returns:
        tuple: (numpy.ndarray (float32), int)
               The loaded audio as a float32 array and the original sample rate.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    audio = np.squeeze(audio)
    return audio, sr

def save_audio(audio, sr, filename="output.wav"):
    """
    Saves a numpy audio array as a WAV file.

    Args:
        audio (numpy.ndarray): The audio data to save.
        sr (int): The sample rate.
        filename (str): The name of the file to save.
    """
    write(filename, sr, audio)  # Save as WAV file
    print(f"Saved as {filename}")
    
# -----------------------------------------
# 2) Audio Pre-processing
# -----------------------------------------

def to_mono_16k(audio, sr, target_sr=16000):
    """
    Downmixes audio to mono and resamples to a target sample rate (default 16k).

    Args:
        audio (numpy.ndarray): The input audio data.
        sr (int): The original sample rate.
        target_sr (int): The target sample rate.

    Returns:
        tuple: (numpy.ndarray (float32), int)
               The processed audio and the target sample rate.
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

# --------------------------------------------
# 3) Real-time Chunking Recorder
# --------------------------------------------

class ChunkRecorder:
    """
    Records from the microphone in real-time and emits fixed-size audio chunks
    via a callback function. Runs in a separate thread.

    Usage:
        def my_callback(chunk):
            print(f"Got chunk of size {len(chunk)}")
        
        rec = ChunkRecorder(on_chunk=my_callback)
        rec.start()
        time.sleep(10)
        rec.stop()
    """
    def __init__(self, sr=16000, channels=1, chunk_size_sec=2.0, blocksize=0, on_chunk=None):
        """
        Initializes the recorder.

        Args:
            sr (int): Sample rate.
            channels (int): Number of channels.
            chunk_size_sec (float): The desired length of each chunk in seconds.
            blocksize (int): The blocksize for the sounddevice stream (0 lets PortAudio choose).
            on_chunk (callable): A function to call with each processed audio chunk.
                                 The chunk will be a 1D numpy.ndarray (float32).
        """
        self.sr = sr
        self.channels = channels
        self.chunk_size_sec = float(chunk_size_sec)
        self.chunk_samples = int(round(self.sr * self.chunk_size_sec))
        if self.chunk_samples <= 0:
            raise ValueError("chunk_size_sec must be > 0")

        self._q = queue.Queue()            # Queue to pass audio blocks from callback to worker
        self._stop_evt = threading.Event() # Event to signal stopping
        self._worker = None                # Worker thread
        self._stream = None                # sounddevice.InputStream

        self._buf = np.empty((0,), dtype=np.float32)  # Internal buffer for audio data
        self.chunks = []                              # Public list to store all chunks
        self.blocksize = blocksize
        self.on_chunk = on_chunk                      # Callback function

    def _callback(self, indata, frames, time_info, status):
        """This is the sounddevice stream callback, running in a separate thread."""
        if status:
            print(f"[audio status] {status}") # non-fatal info (XRuns, etc.)
        
        # Ensure mono float32 shape (samples,)
        x = indata.astype(np.float32)
        if x.ndim == 2 and x.shape[1] > 1:
            x = np.mean(x, axis=1)
        else:
            x = np.squeeze(x)
        self._q.put(x)

    def _worker_loop(self):
        """This is the worker thread loop, consuming the queue."""
        while not self._stop_evt.is_set():
            try:
                # Get a block from the audio callback
                block = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Append to buffer & emit fixed-size chunks
            self._buf = np.concatenate([self._buf, block])
            while len(self._buf) >= self.chunk_samples:
                chunk = self._buf[:self.chunk_samples].copy()
                self.chunks.append(chunk)
                self._buf = self._buf[self.chunk_samples:]
                if self.on_chunk:
                    self.on_chunk(chunk)  # Call the callback function with the chunk

        # After stop event, flush the remainder (if any)
        if len(self._buf) > 0:
            chunk = self._buf.copy()
            self.chunks.append(chunk)
            if self.on_chunk:
                self.on_chunk(chunk)
        self._buf = np.empty((0,), dtype=np.float32)

    def start(self):
        """Starts the recording stream and the worker thread."""
        if self._stream is not None:
            return  # Already running
        
        print(f"[ChunkRecorder] Starting @ {self.sr} Hz, chunk={self.chunk_size_sec}s")
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

    def stop(self):
        """Stops the recording stream and worker thread gracefully."""
        if self._stream is None:
            return
        
        print("[ChunkRecorder] Stopping...")
        self._stop_evt.set()
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
        
        # Wait for worker to finish (flushing remainder)
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        print(f"[ChunkRecorder] Stopped. Total chunks: {len(self.chunks)}")

# -------------------------------
# 4) Test GUI (if run as main)
# -------------------------------
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import messagebox
    from realtime_solo import RealTimeSolo  # Requires realtime_solo.py to be present
    
    class RecordButton:
        """A simple Tkinter GUI to test the ChunkRecorder."""
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

            # Load the ML pipeline for real-time inference
            try:
                self.solo = RealTimeSolo("config.yaml")
                self.solo.start_session()
            except Exception as e:
                messagebox.showerror("Pipeline Error", f"Could not load RealTimeSolo: {e}")
                self.solo = None

        def toggle(self):
            """Toggles the recording state."""
            if not self.is_recording:
                # Start recording
                if self.solo is None:
                    messagebox.showerror("Error", "ML Pipeline not loaded. Cannot start.")
                    return
                try:
                    self.recorder = ChunkRecorder(
                        sr=self.sr, 
                        channels=self.channels, 
                        chunk_size_sec=self.chunk_size_sec, 
                        on_chunk=self.handle_chunk
                    )
                    self.recorder.start()
                    self.is_recording = True
                    self.btn.config(text="Stop")
                except Exception as e:
                    messagebox.showerror("Error", str(e))
            else:
                # Stop recording
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
                        # Save each chunk as output_chunkXX.wav
                        for i, c in enumerate(chunks):
                            fname = f"output_chunk{i+1:02d}.wav"
                            save_audio(c, self.sr, filename=fname)
                            print(f"Saved {fname} ({len(c)/self.sr:.2f}s)")

                    messagebox.showinfo("Done", f"Recorded {len(chunks)} chunks.\nCheck console for details.")

        def handle_chunk(self, chunk):
            """Callback for the recorder, sends chunk to the ML pipeline."""
            if self.solo:
                self.solo.process_chunk(chunk, chunk_sr=self.sr)

    def run_record_button_ui(sr=16000, channels=1, chunk_size_sec=2.0):
        """Initializes and runs the Tkinter test application."""
        root = tk.Tk()
        app = RecordButton(root, sr=sr, channels=channels, chunk_size_sec=chunk_size_sec)
        root.mainloop()
        # Clean up ML session when GUI is closed
        if hasattr(app, 'solo') and app.solo:
            app.solo.stop_session()
            print("ML pipeline session stopped.")
        
    # Launch minimal GUI
    run_record_button_ui(sr=16000, channels=1, chunk_size_sec=2.0)