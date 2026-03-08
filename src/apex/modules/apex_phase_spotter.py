import math
import numpy as np
from typing import List, Tuple, Literal
from scipy.signal import savgol_filter

from src.video.modules import Video
from src.face.modules import FaceLandmark, FaceRoiPoints
from src.optical_flow.modules import TVL1
from src.apex.modules import ApexPhase, ApexSmoother


ExtractionMode = Literal["roi", "fullface"]


class ApexPhaseSpotter:

    def __init__(self,
                 mode: ExtractionMode = "roi",
                 distance_threshold: int = 11,
                 prominence_threshold: float = 0.1,
                 tile_size: Tuple[int, int] = (64, 64),
                 face_size: Tuple[int, int] = (240, 240),
                 margin: float = 0.05):
        """
        Menginisialisasi blueprint untuk deteksi fase apex pada video ekspresi wajah mikro.
        Proses ekstraksi memiliki dua mode yaitu untuk keseluruhan area wajah dan juga fokus pada Region of Interest (RoI) tertentu seperti mata, bibir, dan alis.
        Deteksi fase apex dilakukan dengan menganalisis perubahan magnitudo optical flow menggunakan algoritma TV-L1, kemudian mengidentifikasi titik puncak (apex)
        dan fase berdasarkan kriteria jarak dan prominensi yang ditentukan.

        Args:
            mode (ExtractionMode): Mode ekstraksi yang digunakan, bisa "roi" untuk fokus pada RoI atau "fullface" untuk keseluruhan wajah.
            distance_threshold (int): Batas jarak minimum antara puncak yang terdeteksi untuk dianggap sebagai fase yang berbeda.
            prominence_threshold (float): Batas prominensi minimum untuk puncak yang terdeteksi agar dianggap valid.
            tile_size (Tuple[int, int]): Ukuran target untuk setiap RoI yang diekstraksi dalam mode "roi".
            face_size (Tuple[int, int]): Ukuran target untuk wajah yang diekstraksi dalam mode "fullface".
            margin (float): Margin tambahan yang diterapkan saat mengekstraksi RoI atau wajah untuk memastikan area yang cukup untuk analisis optical flow. 
                            Nilai ini dinyatakan sebagai persentase dari ukuran bounding box yang dihasilkan oleh deteksi landmark wajah.
        """

        self.mode = mode
        self.margin = margin

        self.landmarker = FaceLandmark()

        self.tvl1 = TVL1(fast_mode=True)

        self.apex_phase = ApexPhase(distance_threshold=distance_threshold,
                                    prominence_threshold=prominence_threshold)

        self.tile_w, self.tile_h = tile_size

        self.roi_defs = [frozenset(FaceRoiPoints.LEFT_EYE_POINTS),
                         frozenset(FaceRoiPoints.RIGHT_EYE_POINTS),
                         frozenset(FaceRoiPoints.LIPS_POINTS),
                         frozenset(FaceRoiPoints.LEFT_EYEBROW_POINTS),
                         frozenset(FaceRoiPoints.RIGHT_EYEBROW_POINTS)]

        self.cols = 3
        self.rows = math.ceil(len(self.roi_defs) / self.cols)

        self.face_size = face_size

        self.magnitudes: List[float] = []


    def process(self, video_path: str) -> Tuple[List[int], List[int]]:

        self.magnitudes.clear()

        video = Video(video_path=video_path)

        if self.mode == "roi":
            video.map(self.__process_roi)
        elif self.mode == "fullface":
            video.map(self.__process_fullface)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        return self.__find_apex_phase(self.magnitudes)


    def process_images(self, images: List[np.ndarray]) -> Tuple[List[int], List[int]]:

        self.magnitudes.clear()

        if len(images) < 2:
            return [], []

        prev_frame = images[0]

        for frame_index in range(1, len(images)):
            curr_frame = images[frame_index]

            if self.mode == "roi":
                self.__process_roi(prev_frame, curr_frame, frame_index - 1)
            elif self.mode == "fullface":
                self.__process_fullface(prev_frame, curr_frame, frame_index - 1)
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")

            prev_frame = curr_frame

        return self.__find_apex_phase(self.magnitudes)


    def __process_roi(self,
                      prev_frame: np.ndarray,
                      curr_frame: np.ndarray,
                      frame_index: int) -> None:

        prev_landmarks = self.landmarker.detect(prev_frame)
        curr_landmarks = self.landmarker.detect(curr_frame)

        canvas_mask = np.zeros((self.rows * self.tile_h, self.cols * self.tile_w),dtype=np.uint8)
        canvas_prev = np.zeros((self.rows * self.tile_h, self.cols * self.tile_w, 3), dtype=np.uint8)
        canvas_next = np.zeros_like(canvas_prev)

        for j, roi_points in enumerate(self.roi_defs):

            roi_prev, mask_prev = self.landmarker.crop_roi(image=prev_frame,
                                                           landmark_result=prev_landmarks,
                                                           roi_points=roi_points,
                                                           margin=self.margin,
                                                           target_size=(self.tile_w, self.tile_h))

            roi_next, mask_next = self.landmarker.crop_roi(image=curr_frame,
                                                           landmark_result=curr_landmarks,
                                                           roi_points=roi_points,
                                                           margin=self.margin,
                                                           target_size=(self.tile_w, self.tile_h))

            r, c = divmod(j, self.cols)
            y1, y2 = r * self.tile_h, (r + 1) * self.tile_h
            x1, x2 = c * self.tile_w, (c + 1) * self.tile_w

            canvas_prev[y1:y2, x1:x2] = roi_prev
            canvas_next[y1:y2, x1:x2] = roi_next

            canvas_mask[y1:y2, x1:x2] = ((mask_prev > 0) & (mask_next > 0)).astype(np.uint8)

        flow = self.tvl1.compute(canvas_prev, canvas_next, download=False)

        if hasattr(flow, "download"): flow = flow.download()

        magnitude = np.hypot(flow[..., 0], flow[..., 1])

        valid = canvas_mask > 0
        mean_magnitude = (float(np.mean(magnitude[valid])) if np.any(valid) else float(np.mean(magnitude)))

        self.magnitudes.append(mean_magnitude)


    def __process_fullface(self,
                           prev_frame: np.ndarray,
                           curr_frame: np.ndarray,
                           frame_index: int) -> None:

        prev_landmarks = self.landmarker.detect(prev_frame)
        curr_landmarks = self.landmarker.detect(curr_frame)

        prev_face = self.landmarker.crop(image=prev_frame,
                                         landmarks=prev_landmarks,
                                         margin=self.margin,
                                         output_size=self.face_size)

        curr_face = self.landmarker.crop(image=curr_frame,
                                         landmarks=curr_landmarks,
                                         margin=self.margin,
                                         output_size=self.face_size)

        flow = self.tvl1.compute(prev_face, curr_face, download=False)

        if hasattr(flow, "download"): flow = flow.download()

        magnitude = np.hypot(flow[..., 0], flow[..., 1])

        mean_magnitude = float(np.mean(magnitude))

        self.magnitudes.append(mean_magnitude)


    def __find_apex_phase(self,
                          magnitudes: List[float]) -> Tuple[List[int], List[int]]:

        window_length = ApexSmoother.calculate_window_length(len(magnitudes))
        polyorder = ApexSmoother.calculate_polyorder(window_length)

        smoothed = savgol_filter(magnitudes, window_length, polyorder)

        apex_indices = self.apex_phase.find_apex(signal=smoothed)

        phases = self.apex_phase.find_phase(signal=smoothed, apex_indices=apex_indices)

        return apex_indices, phases
