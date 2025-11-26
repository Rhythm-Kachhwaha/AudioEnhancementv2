"""
Real-time mic enhancement bridge (low-latency, embedding computed once).

Key points:
- Reference speaker embedding is computed ONCE in __init__.
- Real-time audio captured in tiny frames, accumulated, run through model,
  output immediately (no overlap-add to reduce latency).
- MIN_REQUIRED_SAMPLES avoids padding / STFT issues inside the model.
- Supports --profile to see real inference time per chunk.

Run example:
python core\realtime_mic_bridge.py ^
  --checkpoint checkpoint\latest_checkpoint.pt ^
  --reference testing\refaudio\clean_audio_sargam.wav ^
  --chunk 0.16 --input-device 32 --output-device 28 --profile
"""

import argparse
import threading
import queue
import time
import os

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from torch.cuda.amp import autocast

from model import SpeakerAwareEnhancer

SAMPLE_RATE = 16000
MIN_REQUIRED_SAMPLES = 512   # Enough for STFT / model safety


class StreamEnhancer:
    def __init__(
        self, checkpoint_path, reference_audio_path,
        device=None, chunk_seconds=0.1,
        input_device=None, output_device=None,
        profile=False
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.profile = profile
        print(f"[StreamEnhancer] Device: {self.device}")

        # ------------ Load model ------------
        chk = torch.load(checkpoint_path, map_location=self.device)
        self.model = SpeakerAwareEnhancer().to(self.device)

        if "model_state_dict" in chk:
            self.model.load_state_dict(chk["model_state_dict"])
        else:
            self.model.load_state_dict(chk)

        self.model.eval()

        # ------------ Load & embed reference audio ------------
        ref_audio, sr = sf.read(reference_audio_path)
        if ref_audio.ndim > 1:
            ref_audio = np.mean(ref_audio, axis=1)
        if sr != SAMPLE_RATE:
            print(f"[Warning] Reference SR {sr} != target {SAMPLE_RATE}")

        if np.max(np.abs(ref_audio)) > 0:
            ref_audio = ref_audio / np.max(np.abs(ref_audio))

        ref_tensor = torch.from_numpy(ref_audio).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "get_speaker_embeddings"):
                self.speaker_embedding = self.model.get_speaker_embeddings(ref_tensor)
            else:
                print("[Warning] Model lacks get_speaker_embeddings(), using zero embedding.")
                self.speaker_embedding = torch.zeros((1, 256), device=self.device)

        print("[StreamEnhancer] Speaker embedding computed once.")

        # ------------ Chunk size ------------
        self.chunk_seconds = float(chunk_seconds)
        req = int(SAMPLE_RATE * self.chunk_seconds)
        self.chunk_samples = max(req, MIN_REQUIRED_SAMPLES)

        print(f"[StreamEnhancer] Chunk = {self.chunk_samples} samples (~{self.chunk_samples/SAMPLE_RATE:.3f}s)")

        # ------------ Devices ------------
        self.input_device = input_device
        self.output_device = output_device

        # ------------ Queues ------------
        self.input_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=100)

        self.stop_event = threading.Event()

    # ------------------------------------------------------------
    # Audio callback — runs in sounddevice internal thread
    # ------------------------------------------------------------
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio_callback] status: {status}")

        data = indata
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        data = data.astype(np.float32)

        try:
            self.input_queue.put_nowait(data)
        except queue.Full:
            self._drop_in = getattr(self, "_drop_in", 0) + 1
            if self._drop_in % 200 == 0:
                print(f"[audio_callback] dropped {self._drop_in} frames")

    # ------------------------------------------------------------
    # Worker thread — prepares chunks and runs the model
    # ------------------------------------------------------------
    def worker_loop(self):
        pending = np.zeros(0, dtype=np.float32)
        last_log = 0
        is_cuda = self.device == "cuda"

        while not self.stop_event.is_set():
            try:
                hop = self.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            pending = np.concatenate([pending, hop])

            while pending.shape[0] >= self.chunk_samples:
                chunk = pending[:self.chunk_samples]
                pending = pending[self.chunk_samples:]

                # pad for safety
                if chunk.shape[0] < MIN_REQUIRED_SAMPLES:
                    pad = MIN_REQUIRED_SAMPLES - chunk.shape[0]
                    chunk = np.pad(chunk, (0, pad)).astype(np.float32)

                try:
                    noisy_tensor = torch.from_numpy(chunk).float().unsqueeze(0).to(self.device)

                    t0 = time.time()
                    with torch.no_grad():
                        if is_cuda:
                            with autocast():
                                try:
                                    enhanced_tensor, _ = self.model(self.speaker_embedding, noisy_tensor)
                                except TypeError:
                                    enhanced_tensor = self.model(noisy_tensor)
                        else:
                            try:
                                enhanced_tensor, _ = self.model(self.speaker_embedding, noisy_tensor)
                            except TypeError:
                                enhanced_tensor = self.model(noisy_tensor)
                    t1 = time.time()

                    if self.profile:
                        print(f"[profile] infer={1000*(t1-t0):.2f}ms "
                              f"pend={pending.shape[0]} "
                              f"inQ={self.input_queue.qsize()} outQ={self.output_queue.qsize()}")

                    enhanced = enhanced_tensor.squeeze().cpu().numpy().astype(np.float32)

                except Exception as e:
                    now = time.time()
                    if now - last_log > 2:
                        print("[worker] Error:", e)
                        last_log = now
                    enhanced = chunk.copy()

                try:
                    self.output_queue.put_nowait(enhanced)
                except queue.Full:
                    self._drop_out = getattr(self, "_drop_out", 0) + 1
                    if self._drop_out % 100 == 0:
                        print(f"[worker] dropped {self._drop_out} output chunks")

            time.sleep(0.0001)

    # ------------------------------------------------------------
    # Start streaming
    # ------------------------------------------------------------
    def start(self):
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        # Output stream
        self.out_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.output_device,
            blocksize=self.chunk_samples,
        )
        self.out_stream.start()

        # Input stream
        self.in_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.input_device,
            blocksize=256,
            callback=self.audio_callback,
        )
        self.in_stream.start()

        print("[StreamEnhancer] Streaming. Press Ctrl+C to stop.")

        try:
            while True:
                try:
                    frame = self.output_queue.get(timeout=1.0)
                    self.out_stream.write(frame.reshape(-1, 1))
                except queue.Empty:
                    self.out_stream.write(np.zeros((self.chunk_samples, 1), dtype=np.float32))

        except KeyboardInterrupt:
            print("\n[StreamEnhancer] Stopping...")
            self.stop_event.set()
            self.in_stream.stop()
            self.out_stream.stop()
            self.worker_thread.join()
            print("[StreamEnhancer] Stopped.")


def list_audio_devices():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    for i, d in enumerate(devices):
        api = hostapis[d["hostapi"]]["name"]
        print(f"{i:3d} {d['name']} — {api} ({d['max_input_channels']} in, {d['max_output_channels']} out)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--reference", "-r", required=True)
    parser.add_argument("--chunk", type=float, default=0.1)
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--profile", action="store_true")   # <-- FIXED HERE
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    in_dev = int(args.input_device) if args.input_device and args.input_device.isdigit() else args.input_device
    out_dev = int(args.output_device) if args.output_device and args.output_device.isdigit() else args.output_device

    enhancer = StreamEnhancer(
        checkpoint_path=args.checkpoint,
        reference_audio_path=args.reference,
        chunk_seconds=args.chunk,
        input_device=in_dev,
        output_device=out_dev,
        profile=args.profile,
    )
    enhancer.start()


if __name__ == "__main__":
    main()
