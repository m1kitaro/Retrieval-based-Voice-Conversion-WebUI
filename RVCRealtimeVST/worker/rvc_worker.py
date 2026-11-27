"""RVC inference worker for the RVC Realtime VST2/VST3 plugin."""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import struct
import sys
import time
import traceback
from pathlib import Path

MAGIC = 0x50564352
PROTOCOL_VERSION = 2
HEADER_BYTES = 4096
MAX_FRAMES = 1048576
# infer/rtrvc.py keeps a fixed-size pitch cache of this many 10 ms frames.
# The whole inference window has to fit inside it, otherwise net_g.infer
# receives mismatched phone/pitch lengths and raises a size error.
PITCH_CACHE_FRAMES = 1024
MAP_BYTES = HEADER_BYTES + MAX_FRAMES * 4 * 2
INPUT_OFFSET = HEADER_BYTES
OUTPUT_OFFSET = HEADER_BYTES + MAX_FRAMES * 4
STATUS_TEXT_OFFSET = 128
# Request slots written by the plugin for every block.
PITCH_CEILING_OFFSET = 68
PITCH_KNEE_OFFSET = 72
# Response slots read back by the plugin for the VOICE readout.
VOICE_IN_OFFSET = 76
VOICE_OUT_OFFSET = 80
VOICE_LIMITED_OFFSET = 84
STATUS_TEXT_BYTES = 512

STATUS_STARTING = 1
STATUS_LOADING = 2
STATUS_READY = 3
STATUS_ERROR = -1
STATUS_STOP = -2

SYNCHRONIZE = 0x00100000
EVENT_MODIFY_STATE = 0x0002
WAIT_OBJECT_0 = 0
WAIT_TIMEOUT = 258


# A ceiling at or above this is treated as "off": rtrvc already clamps the
# coarse pitch embedding at 1100 Hz, so there is nothing left to limit.
PITCH_CEILING_OFF_HZ = 1100.0
# Rolling window for the VOICE readout (10 ms frames).
VOICE_STATS_FRAMES = 3000
# Frames quieter than this are left out of the readout. rtrvc interpolates F0
# across unvoiced frames, so silence would otherwise drag the median around.
VOICE_STATS_MIN_DBFS = -45.0


def hz_to_midi(hz):
    import numpy as np
    return 12.0 * np.log2(np.asarray(hz, dtype=np.float64) / 440.0) + 69.0


def midi_to_hz(midi):
    import numpy as np
    return 440.0 * np.power(2.0, (np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def soft_ceiling_midi(midi, knee_midi: float, ceiling_midi: float):
    """Leave pitch below the knee untouched and compress everything above it
    so it approaches, but never exceeds, the ceiling. tanh keeps the curve
    continuous with slope 1 at the knee, so there is no audible corner."""
    import numpy as np
    midi = np.asarray(midi, dtype=np.float64)
    width = ceiling_midi - knee_midi
    if width <= 0:
        return np.minimum(midi, ceiling_midi)
    over = midi > knee_midi
    out = midi.copy()
    out[over] = knee_midi + width * np.tanh((midi[over] - knee_midi) / width)
    return out


def write_value(shared: mmap.mmap, offset: int, fmt: str, value) -> None:
    struct.pack_into("<" + fmt, shared, offset, value)


def read_value(shared: mmap.mmap, offset: int, fmt: str):
    return struct.unpack_from("<" + fmt, shared, offset)[0]


def write_status(shared: mmap.mmap, state: int, message: str) -> None:
    write_value(shared, 8, "i", state)
    encoded = message.encode("utf-8", errors="replace")[: STATUS_TEXT_BYTES - 1]
    shared[STATUS_TEXT_OFFSET : STATUS_TEXT_OFFSET + STATUS_TEXT_BYTES] = b"\0" * STATUS_TEXT_BYTES
    shared[STATUS_TEXT_OFFSET : STATUS_TEXT_OFFSET + len(encoded)] = encoded
    shared.flush()


class WinEvent:
    def __init__(self, name: str):
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenEventW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
        kernel32.OpenEventW.restype = ctypes.c_void_p
        self._wait = kernel32.WaitForSingleObject
        self._wait.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._wait.restype = ctypes.c_uint32
        self._set = kernel32.SetEvent
        self._set.argtypes = [ctypes.c_void_p]
        self._set.restype = ctypes.c_int
        self._close = kernel32.CloseHandle
        self._close.argtypes = [ctypes.c_void_p]
        self._close.restype = ctypes.c_int
        self.handle = kernel32.OpenEventW(SYNCHRONIZE | EVENT_MODIFY_STATE, False, name)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), f"OpenEventW failed: {name}")

    def wait(self, timeout_ms: int) -> int:
        return int(self._wait(self.handle, timeout_ms))

    def set(self) -> None:
        if not self._set(self.handle):
            raise OSError(ctypes.get_last_error(), "SetEvent failed")

    def close(self) -> None:
        if self.handle:
            self._close(self.handle)
            self.handle = None


class RVCStreamEngine:
    def __init__(self, cfg: dict):
        self.root = Path(cfg["rvc_root"]).resolve()
        os.chdir(self.root)
        sys.path.insert(0, str(self.root))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "4")

        import librosa
        import numpy as np
        import torch
        import torch.nn.functional as F
        import torchaudio.transforms as tat

        from configs.config import Config
        from infer import rtrvc
        from tools.cuda_graph import run_cuda_graph

        self.librosa = librosa
        self.np = np
        self.torch = torch
        self.F = F
        self.tat = tat
        self.run_cuda_graph = run_cuda_graph

        self.sample_rate = int(cfg["sample_rate"])
        self.block_ms = float(cfg["block_ms"])
        self.crossfade_ms = float(cfg["crossfade_ms"])
        self.extra_ms = float(cfg["extra_ms"])
        self.zc = max(1, self.sample_rate // 100)
        self.block_frame = int(round(self.block_ms / 1000 * self.sample_rate / self.zc) * self.zc)
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = int(round(self.crossfade_ms / 1000 * self.sample_rate / self.zc) * self.zc)
        # Match the source realtime GUI: SOLA overlap is capped at 40 ms and
        # remains independent of the audio callback block length.
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.effective_crossfade_ms = 1000.0 * self.sola_buffer_frame / self.sample_rate
        self.sola_search_frame = self.zc
        self.extra_frame = int(round(self.extra_ms / 1000 * self.sample_rate / self.zc) * self.zc)
        # Keep block + crossfade + SOLA search + context inside the pitch cache.
        window_budget = PITCH_CACHE_FRAMES * self.zc
        fixed_frame = self.block_frame + self.crossfade_frame + self.sola_search_frame
        if fixed_frame > window_budget:
            raise ValueError(
                "block and crossfade exceed the {} ms inference window; reduce Block or Crossfade".format(
                    1000 * window_budget // self.sample_rate
                )
            )
        self.extra_frame = min(self.extra_frame, window_budget - fixed_frame)
        self.effective_extra_ms = 1000.0 * self.extra_frame / self.sample_rate
        self.extra_clamped = self.effective_extra_ms < self.extra_ms - 0.5

        self.config = Config()
        self.rvc = rtrvc.RVC(
            0.0,
            0.0,
            str(Path(cfg["model_path"]).resolve()),
            str(Path(cfg["index_path"]).resolve()) if cfg.get("index_path") else "",
            0.0,
            self.config,
            None,
        )

        total_frames = self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame
        self.input_wav = torch.zeros(total_frames, device=self.config.device, dtype=torch.float32)
        self.input_wav_res = torch.zeros(160 * total_frames // self.zc, device=self.config.device, dtype=torch.float32)
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)
        self.sola_den_kernel = torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device, dtype=torch.float32)
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        self.fade_in_window = torch.sin(
            0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=self.config.device)
        ) ** 2
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(orig_freq=self.sample_rate, new_freq=16000, dtype=torch.float32).to(self.config.device)
        self.resampler2 = None
        if self.rvc.tgt_sr != self.sample_rate:
            self.resampler2 = tat.Resample(orig_freq=self.rvc.tgt_sr, new_freq=self.sample_rate, dtype=torch.float32).to(self.config.device)
        self.last_pitch = None
        self.last_formant = None
        self.last_index_rate = None

        # Pitch limiter. rtrvc multiplies F0 by a fixed transpose; wrap its
        # get_f0 so the shifted contour can be compressed below a ceiling
        # before it reaches the model, without patching the shared RVC code.
        self.pitch_ceiling_hz = 0.0
        self.pitch_knee_st = 5.0
        self.last_f0 = None
        original_get_f0 = self.rvc.get_f0

        def limited_get_f0(x, f0_up_key, method="rmvpe"):
            coarse, f0 = original_get_f0(x, f0_up_key, method)
            limited = self.limit_f0(f0)
            self.last_f0 = (f0, limited, float(f0_up_key))
            if limited is f0:
                return coarse, f0
            return self.rvc.get_f0_post(limited)

        self.rvc.get_f0 = limited_get_f0
        self.reset_voice_stats()

    def reset_voice_stats(self) -> None:
        from collections import deque
        self.voice_in = deque(maxlen=VOICE_STATS_FRAMES)
        self.voice_out = deque(maxlen=VOICE_STATS_FRAMES)
        self.voice_limited = deque(maxlen=VOICE_STATS_FRAMES)
        self.voice_in_hz = 0.0
        self.voice_out_hz = 0.0
        self.voice_limited_pct = 0.0

    def limiter_active(self) -> bool:
        return 0.0 < self.pitch_ceiling_hz < PITCH_CEILING_OFF_HZ

    def limit_f0(self, f0):
        """Apply the soft ceiling to a shifted F0 tensor (Hz, 0 = unvoiced)."""
        if not self.limiter_active():
            return f0
        torch = self.torch
        ceiling_midi = float(hz_to_midi(self.pitch_ceiling_hz))
        knee_midi = ceiling_midi - max(0.1, float(self.pitch_knee_st))
        knee_hz = float(midi_to_hz(knee_midi))
        over = f0 > knee_hz
        if not bool(over.any()):
            return f0
        width = ceiling_midi - knee_midi
        midi = 12.0 * torch.log2(f0[over] / 440.0) + 69.0
        compressed = knee_midi + width * torch.tanh((midi - knee_midi) / width)
        limited = f0.clone()
        limited[over] = 440.0 * torch.pow(2.0, (compressed - 69.0) / 12.0)
        return limited

    def update_voice_stats(self, block_audio) -> None:
        """Track where the input voice and the converted pitch sit, using the
        F0 frames that belong to the block just processed."""
        np = self.np
        if self.last_f0 is None:
            return
        shifted, limited, key = self.last_f0
        frames = self.block_frame // self.zc
        if frames <= 0:
            return
        shifted = shifted.detach().float().cpu().numpy().reshape(-1)
        limited = limited.detach().float().cpu().numpy().reshape(-1)
        # rtrvc keeps pitch[3:-1] of every call, so the newest usable frames
        # end one frame before the end of the extracted contour.
        tail = slice(-(frames + 1), -1)
        shifted = shifted[tail]
        limited = limited[tail]
        audio = np.asarray(block_audio, dtype=np.float32)[: frames * self.zc]
        if audio.size < frames * self.zc or shifted.size != frames:
            return
        rms = np.sqrt(np.mean(audio.reshape(frames, self.zc) ** 2, axis=1) + 1e-12)
        voiced = (20.0 * np.log10(rms) > VOICE_STATS_MIN_DBFS) & (shifted > 0)
        if not voiced.any():
            return
        self.voice_in.extend((shifted[voiced] / 2.0 ** (key / 12.0)).tolist())
        self.voice_out.extend(limited[voiced].tolist())
        self.voice_limited.extend((limited[voiced] < shifted[voiced] * 2.0 ** (-0.5 / 12.0)).tolist())
        self.voice_in_hz = float(np.median(self.voice_in))
        self.voice_out_hz = float(np.median(self.voice_out))
        self.voice_limited_pct = 100.0 * float(np.mean(self.voice_limited))

    def prewarm(self) -> None:
        phase = self.torch.arange(self.block_frame, device=self.config.device, dtype=self.torch.float32)
        probe = 0.05 * self.torch.sin(2 * self.np.pi * 220.0 * phase / self.sample_rate)
        self.process(probe.cpu().numpy(), 12.0, 0.0, 0.0, 0.5, -60.0, 0, 0.0, 5.0)
        if self.torch.device(self.config.device).type == "cuda":
            self.torch.cuda.synchronize(self.config.device)

        self.input_wav.zero_()
        self.input_wav_res.zero_()
        self.rms_buffer.fill(0.0)
        self.sola_buffer.zero_()
        if hasattr(self.rvc, "cache_pitch"):
            self.rvc.cache_pitch.zero_()
        if hasattr(self.rvc, "cache_pitchf"):
            self.rvc.cache_pitchf.zero_()
        self.last_f0 = None
        self.reset_voice_stats()

    def process(self, audio, pitch: float, formant: float, index_rate: float,
                rms_mix: float, threshold: float, f0_method: int,
                pitch_ceiling_hz: float = 0.0, pitch_knee_st: float = 5.0):
        np = self.np
        torch = self.torch
        F = self.F
        if len(audio) != self.block_frame:
            raise ValueError(f"block mismatch: worker={self.block_frame}, request={len(audio)}")

        if self.last_pitch != pitch:
            self.rvc.change_key(pitch)
            self.last_pitch = pitch
        if self.last_formant != formant:
            self.rvc.change_formant(formant)
            self.last_formant = formant
        if self.last_index_rate != index_rate:
            self.rvc.change_index_rate(index_rate)
            self.last_index_rate = index_rate

        self.pitch_ceiling_hz = float(pitch_ceiling_hz)
        self.pitch_knee_st = float(pitch_knee_st)
        # The readout is a rolling median, so frames converted with earlier
        # settings would keep skewing it for a long time. Start over whenever
        # anything that moves the output pitch changes.
        settings = (float(pitch) - float(formant), self.pitch_ceiling_hz, self.pitch_knee_st)
        if settings != getattr(self, "voice_settings", None):
            self.voice_settings = settings
            self.reset_voice_stats()

        indata = np.asarray(audio, dtype=np.float32).copy()
        raw_block = indata.copy()
        if threshold > -60.0:
            gated = np.append(self.rms_buffer, indata)
            rms = self.librosa.feature.rms(y=gated, frame_length=4 * self.zc, hop_length=self.zc)[:, 2:]
            self.rms_buffer[:] = gated[-4 * self.zc :]
            gated = gated[2 * self.zc - self.zc // 2 :]
            below = self.librosa.amplitude_to_db(rms, ref=1.0)[0] < threshold
            for i, mute in enumerate(below):
                if mute:
                    gated[i * self.zc : (i + 1) * self.zc] = 0
            indata = gated[self.zc // 2 :]

        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-self.block_frame:] = torch.from_numpy(indata).to(self.config.device)
        self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].clone()
        resample_input = self.input_wav[-self.block_frame - 2 * self.zc :]
        resampled = self.run_cuda_graph(
            self.resampler,
            "vst-input-resample",
            lambda value: self.resampler(value),
            resample_input,
        )[160:]
        self.input_wav_res[-self.block_frame_16k:] = resampled[-self.block_frame_16k:]

        method = ("rmvpe", "fcpe", "pm")[max(0, min(2, int(f0_method)))]
        infer_wav = self.rvc.infer(
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.return_length,
            method,
        )
        self.update_voice_stats(raw_block)
        if self.resampler2 is not None:
            infer_wav = self.run_cuda_graph(
                self.resampler2,
                "vst-output-resample",
                lambda value: self.resampler2(value),
                infer_wav,
            )

        if rms_mix < 1.0:
            input_tail = self.input_wav[self.extra_frame :]
            rms1 = self.librosa.feature.rms(
                y=input_tail[: infer_wav.shape[0]].cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc
            )
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(rms1.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = self.librosa.feature.rms(
                y=infer_wav.cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc
            )
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(rms2.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = torch.maximum(rms2, torch.full_like(rms2, 1e-3))
            infer_wav *= torch.pow(rms1 / rms2, 1.0 - rms_mix)

        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input**2, self.sola_den_kernel) + 1e-8)
        sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]))
        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
        return infer_wav[: self.block_frame].float().cpu().numpy()


def run(args: argparse.Namespace) -> int:
    shared = None
    request_event = None
    response_event = None
    try:
        shared = mmap.mmap(-1, MAP_BYTES, tagname=args.map, access=mmap.ACCESS_WRITE)
        request_event = WinEvent(args.request)
        response_event = WinEvent(args.response)
        if read_value(shared, 0, "I") != MAGIC or read_value(shared, 4, "I") != PROTOCOL_VERSION:
            raise RuntimeError("RVC VST protocol mismatch")
        write_status(shared, STATUS_LOADING, "Loading Python and CUDA")
        with open(args.config, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        # RVC's Config parses process-wide CLI flags intended for its WebUI.
        sys.argv = [sys.argv[0]]
        engine = RVCStreamEngine(cfg)
        write_status(shared, STATUS_LOADING, "Prewarming CUDA and F0")
        engine.prewarm()
        ready_text = f"Ready (actual CF {engine.effective_crossfade_ms:.0f} ms"
        if engine.extra_clamped:
            ready_text += f", context capped at {engine.effective_extra_ms:.0f} ms"
        write_status(shared, STATUS_READY, ready_text + ")")

        import numpy as np

        input_view = np.ndarray((MAX_FRAMES,), dtype=np.float32, buffer=shared, offset=INPUT_OFFSET)
        output_view = np.ndarray((MAX_FRAMES,), dtype=np.float32, buffer=shared, offset=OUTPUT_OFFSET)
        last_sequence = 0
        while True:
            result = request_event.wait(1000)
            if result == WAIT_TIMEOUT:
                continue
            if result != WAIT_OBJECT_0:
                raise OSError("WaitForSingleObject failed")
            if read_value(shared, 8, "i") == STATUS_STOP:
                break
            sequence = read_value(shared, 12, "I")
            if sequence == last_sequence:
                continue
            frames = read_value(shared, 20, "I")
            if frames <= 0 or frames > MAX_FRAMES:
                raise ValueError(f"invalid frame count: {frames}")
            started = time.perf_counter()
            processed = engine.process(
                input_view[:frames],
                read_value(shared, 32, "f"),
                read_value(shared, 36, "f"),
                read_value(shared, 40, "f"),
                read_value(shared, 44, "f"),
                read_value(shared, 48, "f"),
                read_value(shared, 64, "I"),
                read_value(shared, PITCH_CEILING_OFFSET, "f"),
                read_value(shared, PITCH_KNEE_OFFSET, "f"),
            )
            output_view[:frames] = processed
            write_value(shared, VOICE_IN_OFFSET, "f", engine.voice_in_hz)
            write_value(shared, VOICE_OUT_OFFSET, "f", engine.voice_out_hz)
            write_value(shared, VOICE_LIMITED_OFFSET, "f", engine.voice_limited_pct)
            write_value(shared, 56, "f", (time.perf_counter() - started) * 1000.0)
            write_value(shared, 16, "I", sequence)
            write_value(shared, 8, "i", STATUS_READY)
            last_sequence = sequence
            response_event.set()
        return 0
    except BaseException:
        message = traceback.format_exc()
        try:
            Path(args.config + ".log").write_text(message, encoding="utf-8")
        except Exception:
            pass
        try:
            if shared is not None:
                write_status(shared, STATUS_ERROR, message[-500:])
            if response_event is not None:
                response_event.set()
        except Exception:
            pass
        return 1
    finally:
        if request_event is not None:
            request_event.close()
        if response_event is not None:
            response_event.close()
        if shared is not None:
            shared.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
