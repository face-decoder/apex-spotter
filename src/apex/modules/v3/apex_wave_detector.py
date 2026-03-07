import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict


class ApexWaveDetector:
    """Deteksi gelombang (wave) pada sinyal menggunakan valley-to-valley segmentation.

    Pendekatan ini menggantikan peak-based detection dengan terlebih dahulu
    mencari valley (lembah), lalu mensegmentasi gelombang antar dua valley
    berurutan. Apex ditentukan sebagai titik tertinggi dalam setiap segmen.

    Pipeline:
        Signal → Valley Detection → Wave Segmentation → Apex Detection (per wave)
    """

    def detect_waves(self, signal: list) -> List[Dict[str, int]]:
        """Segmentasi sinyal menjadi gelombang berdasarkan valley-to-valley.

        Setiap gelombang memiliki onset (valley kiri), offset (valley kanan),
        dan apex (titik tertinggi dalam segmen tersebut).

        Args:
            signal: Sinyal input (magnitude optical flow yang sudah di-smooth).

        Returns:
            List of dict, masing-masing berisi:
                - "onset":  indeks valley kiri (awal gelombang)
                - "apex":   indeks titik tertinggi dalam gelombang
                - "offset": indeks valley kanan (akhir gelombang)
        """
        signal = np.array(signal)

        # Cari valley (local minima) dengan membalik sinyal
        valleys, _ = find_peaks(-signal)

        waves: List[Dict[str, int]] = []

        for i in range(len(valleys) - 1):
            onset = int(valleys[i])
            offset = int(valleys[i + 1])

            segment = signal[onset:offset]

            # Skip segmen yang terlalu pendek untuk memiliki puncak valid
            if len(segment) < 3:
                continue

            apex = onset + int(np.argmax(segment))

            waves.append({
                "onset": onset,
                "apex": apex,
                "offset": offset,
            })

        return waves
