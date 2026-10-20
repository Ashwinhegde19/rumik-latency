"""Objective signal metrics for 24 kHz mono TTS clips (shared by the WAV analysis and the paired stats).

All metrics work on float audio in [-1, 1]. One Mimi codec frame is 1920 samples (80 ms at 24 kHz).
"""
from __future__ import annotations

import numpy as np

SR = 24000
FRAME = 1920          # samples per codec frame (80 ms)
SILENCE_DBFS = -40.0  # same threshold the benchmarks use


def db(x, floor=1e-9):
    return 20.0 * np.log10(np.maximum(x, floor))


def load_wav(path: str):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    info = sf.info(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr, info


def grid_frame_dbfs(x: np.ndarray, frame: int = FRAME) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.array([])
    f = x[: n * frame].reshape(n, frame).astype(np.float64)
    return db(np.sqrt((f ** 2).mean(axis=1)))


def short_dbfs(x: np.ndarray, sr: int = SR, win_ms: float = 20.0, hop_ms: float = 10.0) -> np.ndarray:
    w, h = int(sr * win_ms / 1000), int(sr * hop_ms / 1000)
    if len(x) < w:
        return np.array([db(np.sqrt(np.mean(x.astype(np.float64) ** 2)))])
    n = 1 + (len(x) - w) // h
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    seg = x[idx].astype(np.float64)
    return db(np.sqrt((seg ** 2).mean(axis=1)))


def silence_edges(x: np.ndarray, sr: int = SR, thr: float = SILENCE_DBFS, hop_ms: float = 10.0):
    """Leading / trailing silence in ms (20 ms windows, 10 ms hop, threshold thr dBFS)."""
    d = short_dbfs(x, sr, 20.0, hop_ms)
    act = np.where(d > thr)[0]
    if len(act) == 0:
        return len(x) / sr * 1000, len(x) / sr * 1000
    lead = act[0] * hop_ms
    end_ms = act[-1] * hop_ms + 20.0
    trail = max(0.0, len(x) / sr * 1000 - end_ms)
    return float(lead), float(trail)


def spectral_stats(x: np.ndarray, sr: int = SR, active_thr: float = SILENCE_DBFS):
    """Mean spectral centroid / flatness / >6 kHz energy ratio over speech-active STFT frames."""
    n_fft, hop = 1024, 240
    if len(x) < n_fft:
        return {"centroid_hz": None, "flatness": None, "hf_ratio_6k": None}
    win = np.hanning(n_fft)
    n = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    seg = x[idx].astype(np.float64)
    rms_db = db(np.sqrt((seg ** 2).mean(axis=1)))
    keep = rms_db > active_thr
    if keep.sum() == 0:
        return {"centroid_hz": None, "flatness": None, "hf_ratio_6k": None}
    S = np.abs(np.fft.rfft(seg[keep] * win, axis=1)) ** 2 + 1e-20
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    centroid = (S * freqs).sum(1) / S.sum(1)
    flat = np.exp(np.log(S).mean(1)) / S.mean(1)
    hf = S[:, freqs >= 6000].sum(1) / S.sum(1)
    return {"centroid_hz": float(centroid.mean()), "flatness": float(flat.mean()), "hf_ratio_6k": float(hf.mean())}


def phase_profile(x: np.ndarray, period: int = FRAME, order: int = 1, active_only: bool = True):
    """Mean |difference| of the waveform at each phase modulo `period`.

    A discontinuity at codec-frame boundaries shows up as an outlier at phase 0 (first difference
    x[n]-x[n-1] across the boundary) or phases period-1 / 0 (second difference)."""
    x = x.astype(np.float64)
    if order == 1:
        d = np.zeros_like(x)
        d[1:] = x[1:] - x[:-1]
    else:
        d = np.zeros_like(x)
        d[1:-1] = x[2:] - 2 * x[1:-1] + x[:-2]
    n = len(x) // period
    if n < 2:
        return None, 0
    D = np.abs(d[: n * period].reshape(n, period))
    if active_only:
        fd = grid_frame_dbfs(x, period)
        keep = fd[:n] > SILENCE_DBFS
        # need the previous frame active too for a boundary at the start of this frame to be audible
        keep[1:] &= fd[: n - 1] > SILENCE_DBFS
        keep[0] = False
        if keep.sum() < 2:
            return None, 0
        D = D[keep]
    return D.sum(0), D.shape[0]


def phase_profile_norm(x: np.ndarray, period: int = FRAME, win: int = 241):
    """Like phase_profile(order=1) but each |x[n]-x[n-1]| is divided by its local mean (+-win/2 samples),
    which removes the speech envelope and makes a fixed-size boundary step much easier to see."""
    from scipy.ndimage import uniform_filter1d
    x = x.astype(np.float64)
    d = np.zeros_like(x)
    d[1:] = np.abs(x[1:] - x[:-1])
    loc = uniform_filter1d(d, size=win, mode="nearest") + 1e-7
    r = d / loc
    n = len(x) // period
    if n < 2:
        return None, 0
    R = r[: n * period].reshape(n, period)
    fd = grid_frame_dbfs(x, period)
    keep = fd[:n] > SILENCE_DBFS
    keep[1:] &= fd[: n - 1] > SILENCE_DBFS
    keep[0] = False
    if keep.sum() < 2:
        return None, 0
    R = R[keep]
    return R.sum(0), R.shape[0]


def hf_energy_fold(x: np.ndarray, sr: int = SR, period: int = FRAME, bin_len: int = 24, fc: float = 8000.0):
    """High-passed (>fc) energy in 1 ms bins, each frame normalised to mean 1, folded modulo `period`.
    Periodic clicks at frame boundaries would raise the first / last bin."""
    from scipy.signal import butter, sosfilt
    sos = butter(6, fc, btype="highpass", fs=sr, output="sos")
    y = sosfilt(sos, x.astype(np.float64))
    n = len(y) // period
    if n < 2:
        return None, 0
    E = (y[: n * period] ** 2).reshape(n, period // bin_len, bin_len).sum(-1)
    fd = grid_frame_dbfs(x, period)
    keep = (fd[:n] > SILENCE_DBFS)
    keep[1:] &= fd[: n - 1] > SILENCE_DBFS
    keep[0] = False
    E = E[keep]
    if E.shape[0] < 2:
        return None, 0
    E = E / (E.mean(1, keepdims=True) + 1e-20)
    return E.sum(0), E.shape[0]


def boundary_z(profile_sum: np.ndarray, count: int, phases=(0,)):
    """z-score and percentile rank of the boundary phase(s) relative to all other phases."""
    if profile_sum is None or count == 0:
        return None
    m = profile_sum / count
    mask = np.ones(len(m), bool)
    for p in phases:
        mask[p % len(m)] = False
    rest = m[mask]
    b = max(m[p % len(m)] for p in phases)
    return {"boundary_mean": float(b), "other_mean": float(rest.mean()), "ratio": float(b / rest.mean()),
            "z": float((b - rest.mean()) / rest.std()), "pct_rank": float((rest < b).mean())}


def clip_metrics(x: np.ndarray, sr: int = SR) -> dict:
    fd = grid_frame_dbfs(x)
    act = np.where(fd > SILENCE_DBFS)[0]
    lead_ms, trail_ms = silence_edges(x, sr)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    active = x[: len(fd) * FRAME].reshape(len(fd), FRAME)[fd > SILENCE_DBFS] if len(act) else x[:0]
    out = {
        "duration_s": len(x) / sr,
        "codec_frames": len(x) / FRAME,
        "peak": peak,
        "peak_dbfs": float(db(peak)),
        "n_clip": int((np.abs(x) >= 32766 / 32768).sum()),
        "n_over_0p99": int((np.abs(x) >= 0.99).sum()),
        "rms_dbfs": float(db(np.sqrt(np.mean(x.astype(np.float64) ** 2)))) if len(x) else None,
        "active_rms_dbfs": float(db(np.sqrt(np.mean(active.astype(np.float64) ** 2)))) if active.size else None,
        "first_nonsilent_frame": int(act[0]) if len(act) else None,
        "last_nonsilent_frame": int(act[-1]) if len(act) else None,
        "trailing_silent_frames": int(len(fd) - 1 - act[-1]) if len(act) else None,
        "leading_silence_ms": lead_ms,
        "trailing_silence_ms": trail_ms,
        "speech_frames": int(len(act)),
    }
    out.update(spectral_stats(x, sr))
    for per, name in ((FRAME, "f80"), (FRAME // 2, "f40")):
        p1, c1 = phase_profile(x, per, 1)
        p2, c2 = phase_profile(x, per, 2)
        pn, cn = phase_profile_norm(x, per)
        ph, ch = hf_energy_fold(x, SR, per)
        out[f"bnd_{name}_d1"] = boundary_z(p1, c1, (0,))
        out[f"bnd_{name}_d2"] = boundary_z(p2, c2, (per - 1, 0))
        out[f"bnd_{name}_d1norm"] = boundary_z(pn, cn, (0,))
        out[f"bnd_{name}_hf"] = boundary_z(ph, ch, (0, -1))
    return out


def snr_db(ref: np.ndarray, x: np.ndarray) -> float:
    n = min(len(ref), len(x))
    e = x[:n].astype(np.float64) - ref[:n].astype(np.float64)
    return float(10 * np.log10(np.sum(ref[:n].astype(np.float64) ** 2) / max(np.sum(e ** 2), 1e-30)))
