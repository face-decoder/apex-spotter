import numpy as np
from scipy.signal import find_peaks


class ApexPhase:

    # Ambang batas yang digunakan untuk menentukan jarak antar titik puncak
    DISTANCE_THRESHOLD = 5

    # Ambang batas yang digunakan untuk menentukan keunggulan puncak
    PROMINENCE_THRESHOLD = 0.01

    # Radius pencarian maksimal dari apex untuk onset/offset
    MAX_SEARCH_RADIUS = 100

    def __init__(self,
                 distance_threshold: int = DISTANCE_THRESHOLD,
                 prominence_threshold: float = PROMINENCE_THRESHOLD) -> None:

        self.distance = distance_threshold
        self.prominence = prominence_threshold


    def find_apex(self, signal: list, height: float = None) -> list:
        """
        Mendeteksi titik puncak (apex) dalam sinyal menggunakan metode find_peaks dari scipy.

        Args:
            signal (list): Sinyal input yang akan dianalisis.
            height (float): Ambang batas tinggi minimum untuk peak.
                            Jika None, tidak ada filter tinggi.

        Returns:
            list: Indeks titik puncak yang terdeteksi dalam sinyal.
        """
        kwargs = dict(distance=self.distance, prominence=self.prominence)
        if height is not None:
            kwargs['height'] = height

        peaks, _ = find_peaks(signal, **kwargs)
        return peaks.tolist()


    def find_top_k_apex(self, signal: list, k: int = 3, height: float = None) -> list:
        """
        Mendeteksi top-K titik puncak berdasarkan prominence tertinggi.

        Args:
            signal (list): Sinyal input yang akan dianalisis.
            k (int): Jumlah maksimal apex yang dikembalikan.
            height (float): Ambang batas tinggi minimum untuk peak.

        Returns:
            list: Indeks top-K titik puncak, diurutkan secara ascending.
        """
        kwargs = dict(distance=self.distance, prominence=self.prominence)
        if height is not None:
            kwargs['height'] = height

        peaks, properties = find_peaks(signal, **kwargs)

        if len(peaks) == 0:
            return []

        if len(peaks) <= k:
            return peaks.tolist()

        # Ambil top-K berdasarkan prominence tertinggi
        prominences = properties['prominences']
        top_indices = np.argsort(prominences)[::-1][:k]
        top_peaks = np.sort(peaks[top_indices])
        return top_peaks.tolist()


    def find_phase(self, signal: list, apex_indices: list) -> dict:
        """
        Mendeteksi fase apex berdasarkan sinyal dan indeks apex yang sudah ditemukan.

        Metode ini hanya mencari local valley di kiri/kanan dari apex.

        Args:
            signal (list): Sinyal input yang akan diproses.
            apex_indices (list): Daftar indeks apex yang sudah ditemukan.

        Returns:
            dict: Kamus yang berisi informasi fase apex.
        """
        phases = dict()

        for idx, apex_index in enumerate(apex_indices):

            # Midpoint boundary (mencegah tumpang tindih antar fase)
            left_bound = 0 if idx == 0 else (apex_indices[idx - 1] + apex_index) // 2
            right_bound = len(signal) - 1 if idx == len(apex_indices) - 1 else (apex_index + apex_indices[idx + 1]) // 2

            start_index, end_index = self.__find_phase_boundaries(signal=signal,
                                                                  apex_index=apex_index,
                                                                  left_bound=left_bound,
                                                                  right_bound=right_bound)

            phases[apex_index] = dict(start=start_index, end=end_index)

        return phases


    def __find_phase_boundaries(self,
                                signal: list,
                                apex_index: int,
                                left_bound: int = 0,
                                right_bound: int = None) -> tuple:
        """
        Mendeteksi batas fase apex dengan mencari local valley (titik terendah lokal) 
        kiri/kanan dari apex, tanpa menggunakan cutoff dan threshold.

        Args:
            signal (list): Sinyal input yang akan dianalisis.
            apex_index (int): Indeks titik apex dalam sinyal.
            left_bound (int): Batas kiri pencarian (mencegah tumpang tindih).
            right_bound (int): Batas kanan pencarian (mencegah tumpang tindih).

        Returns:
            tuple: Indeks batas awal dan akhir fase apex (valleys).
        """
        if right_bound is None:
            right_bound = len(signal) - 1

        # Batasi search radius agar tidak terlalu lebar
        effective_left = max(left_bound, apex_index - self.MAX_SEARCH_RADIUS)
        effective_right = min(right_bound, apex_index + self.MAX_SEARCH_RADIUS)

        # ── Cari local valley kiri (turun dari apex sampai naik lagi) ──
        valley_left = apex_index
        for i in range(apex_index - 1, effective_left - 1, -1):
            if signal[i] > signal[i + 1]:
                # Sinyal mulai naik → valley ditemukan di i+1
                valley_left = i + 1
                break
            valley_left = i

        # ── Cari local valley kanan (turun dari apex sampai naik lagi) ──
        valley_right = apex_index
        for i in range(apex_index + 1, effective_right + 1):
            if signal[i] > signal[i - 1]:
                # Sinyal mulai naik → valley ditemukan di i-1
                valley_right = i - 1
                break
            valley_right = i

        return valley_left, valley_right