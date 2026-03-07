import numpy as np
from typing import List, Dict


class ApexWaveValidator:
    """Validasi gelombang hasil segmentasi untuk menyaring noise dan false positives.

    Tiga kriteria filtering diterapkan:
        1. Amplitude  — tolak gelombang dengan amplitudo terlalu kecil
        2. Duration   — tolak gelombang terlalu pendek atau terlalu panjang
        3. Symmetry   — tolak gelombang yang sangat asimetris (rise vs fall)
    """

    # Default thresholds
    DEFAULT_AMPLITUDE_THRESHOLD = 0.01
    DEFAULT_MIN_DURATION = 3
    DEFAULT_MAX_DURATION = 200
    DEFAULT_SYMMETRY_MIN = 0.3
    DEFAULT_SYMMETRY_MAX = 3.0

    def __init__(self,
                 amplitude_threshold: float = DEFAULT_AMPLITUDE_THRESHOLD,
                 min_duration: int = DEFAULT_MIN_DURATION,
                 max_duration: int = DEFAULT_MAX_DURATION,
                 symmetry_min: float = DEFAULT_SYMMETRY_MIN,
                 symmetry_max: float = DEFAULT_SYMMETRY_MAX) -> None:
        """
        Args:
            amplitude_threshold: Ambang batas minimum amplitude gelombang.
            min_duration: Durasi minimum gelombang (dalam jumlah frame).
            max_duration: Durasi maksimum gelombang (dalam jumlah frame).
            symmetry_min: Batas bawah rasio rise/fall untuk simetri.
            symmetry_max: Batas atas rasio rise/fall untuk simetri.
        """
        self.amplitude_threshold = amplitude_threshold
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.symmetry_min = symmetry_min
        self.symmetry_max = symmetry_max

    def validate(self,
                 waves: List[Dict[str, int]],
                 signal: list) -> List[Dict[str, int]]:
        """Filter gelombang berdasarkan amplitude, durasi, dan simetri.

        Args:
            waves: List gelombang dari ApexWaveDetector.detect_waves().
            signal: Sinyal asli (untuk menghitung amplitude).

        Returns:
            List gelombang yang lolos semua kriteria validasi.
        """
        signal = np.array(signal)
        validated: List[Dict[str, int]] = []

        for wave in waves:
            onset = wave["onset"]
            apex = wave["apex"]
            offset = wave["offset"]

            # ── 1. Amplitude filtering ──
            amplitude = signal[apex] - min(signal[onset], signal[offset])
            if amplitude < self.amplitude_threshold:
                continue

            # ── 2. Duration filtering ──
            duration = offset - onset
            if duration < self.min_duration:
                continue
            if duration > self.max_duration:
                continue

            # ── 3. Symmetry filtering ──
            rise = apex - onset
            fall = offset - apex

            if fall == 0:
                continue

            ratio = rise / fall
            if ratio < self.symmetry_min or ratio > self.symmetry_max:
                continue

            validated.append(wave)

        return validated
