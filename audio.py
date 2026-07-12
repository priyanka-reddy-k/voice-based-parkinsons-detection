import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

def extract_vocal_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    if len(y) > 0:
        y = librosa.util.normalize(y)
        
    pitch_floor, pitch_ceil = 75.0, 600.0

    # ── PITCH FEATURES (Fo, Fhi, Flo) ──────────────────────────
    pitches, _ = librosa.piptrack(y=y, sr=sr, fmin=75.0, fmax=300.0)
    valid_pitches = pitches[(pitches >= 75.0) & (pitches <= 300.0)]
    
    mean_pitch = np.mean(valid_pitches) if len(valid_pitches) > 0 else 122.0
    fhi        = np.max(valid_pitches)  if len(valid_pitches) > 0 else 150.0
    flo        = np.min(valid_pitches)  if len(valid_pitches) > 0 else 100.0

    # ── PRAAT SOUND OBJECT ──────────────────────────────────────
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    point_process = call(sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceil)

    # ── JITTER FEATURES (local %, absolute, RAP) ────────────────
    local_jitter     = call(point_process, "Get jitter (local)",          0.0, 0.0, 0.0001, 0.02, 1.3)
    jitter_absolute  = call(point_process, "Get jitter (local, absolute)", 0.0, 0.0, 0.0001, 0.02, 1.3)
    rap              = call(point_process, "Get jitter (rap)",             0.0, 0.0, 0.0001, 0.02, 1.3)

    local_jitter    = local_jitter    if not np.isnan(local_jitter)    else 0.004
    jitter_absolute = jitter_absolute if not np.isnan(jitter_absolute) else 0.00004
    rap             = rap             if not np.isnan(rap)             else 0.002

    # ── SHIMMER FEATURES (local %, APQ3) ────────────────────────
    local_shimmer = call([sound, point_process], "Get shimmer (local)",    0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
    apq3          = call([sound, point_process], "Get shimmer (apq3)",     0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)

    local_shimmer = local_shimmer if not np.isnan(local_shimmer) else 0.025
    apq3          = apq3          if not np.isnan(apq3)          else 0.013

    # ── NOISE RATIOS (HNR, NHR) ─────────────────────────────────
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, pitch_floor, 0.1, 4.5)
    hnr         = call(harmonicity, "Get mean", 0.0, 0.0)
    hnr         = hnr if not np.isnan(hnr) else 21.5
    nhr         = 1.0 / (10 ** (hnr / 10)) if hnr > 0 else 0.05  # derived from HNR

    # ── UNIT CONVERSIONS ────────────────────────────────────────
    jitter_percent  = local_jitter  * 100
    shimmer_percent = local_shimmer * 100

    # ── FEATURE VECTOR (must match engine.py order exactly) ─────
    features = [
        mean_pitch,       # MDVP:Fo(Hz)
        fhi,              # MDVP:Fhi(Hz)
        flo,              # MDVP:Flo(Hz)
        jitter_percent,   # MDVP:Jitter(%)
        jitter_absolute,  # MDVP:Jitter(Abs)
        rap,              # MDVP:RAP
        shimmer_percent,  # MDVP:Shimmer
        apq3,             # Shimmer:APQ3
        hnr,              # HNR
        nhr               # NHR
    ]

    return np.array(features).reshape(1, -1), {
        "Mean Pitch (Hz)":    mean_pitch,
        "Local Jitter (%)":   jitter_percent,
        "Local Shimmer (%)":  shimmer_percent,
        "HNR (Clarity dB)":   hnr
    }