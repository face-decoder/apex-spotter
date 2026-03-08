import cv2
import numpy as np
from typing import List, Literal

from src.video.modules import Video
from src.face.modules import FaceLandmark, FaceRoiPoints, FaceRoiSizes
from src.optical_flow.modules import TVL1
from src.apex.modules.v2 import ApexPhase, ApexSmoother


class ApexPhaseSpotter:

    def __init__(self,
                 margin: float = 0.05,
                 mode: Literal["single", "batch"] = "single"):
        """
        Apex phase detector berbasis ROI.

        Args:
            margin: Margin tambahan saat cropping ROI dari landmark.
            mode:   "single" optical flow dihitung per ROI secara independen.
                    "batch"  semua ROI digabung (vstack) dan dihitung sekaligus.
        """

        if mode not in ("single", "batch"):
            raise ValueError(f"Invalid mode '{mode}'. Expected 'single' or 'batch'.")

        self.margin = margin
        self.mode = mode

        self.landmarker = FaceLandmark()

        self.tvl1 = TVL1(fast_mode=True)

        self.apex_phase = ApexPhase(cutoff_ratio=0.35,
                                    distance_threshold=30,
                                    prominence_threshold=0.05)

        self.roi_defs = [
            (FaceRoiPoints.LEFT_EYE_POINTS,    FaceRoiSizes.EYE_SIZE),
            (FaceRoiPoints.RIGHT_EYE_POINTS,   FaceRoiSizes.EYE_SIZE),
            (FaceRoiPoints.LIPS_POINTS,        FaceRoiSizes.LIPS_SIZE),
            (FaceRoiPoints.LEFT_EYEBROW_POINTS,  FaceRoiSizes.EYEBROW_SIZE),
            (FaceRoiPoints.RIGHT_EYEBROW_POINTS, FaceRoiSizes.EYEBROW_SIZE),
        ]

        self.magnitudes: List[float] = []
        self.horizontal_magnitudes: List[float] = []
        self.vertical_magnitudes: List[float] = []

        self.horizontal_vector: np.ndarray = np.array([1, 0], dtype=np.float32)

        self.vertical_vector: np.ndarray = np.array([0, 1], dtype=np.float32)

        # Cache landmark frame sebelumnya agar tidak ada deteksi ganda.
        # curr_landmarks iterasi N menjadi prev_landmarks iterasi N+1.
        self._cached_landmarks = None



    def process(self, video_path: str):
        """
        Proses satu video dan kembalikan apex indices beserta phases.
        
        Args:
            video_path: Path ke file video yang akan diproses.

        Returns:
            Tuple (apex_indices, phases) hasil deteksi apex phase pada video.
        """

        self._reset()

        video = Video(video_path=video_path)
        video.map(self.__process_roi)

        return self.__find_apex_phase(self.magnitudes)


    def process_videos(self, video_paths: List[str]):
        """
        Proses beberapa video secara berurutan dan kembalikan apex indices beserta phases.
        
        Args:
            video_paths: List of paths ke file video yang akan diproses.

        Returns:
            Tuple (apex_indices, phases) hasil deteksi apex phase pada semua video.
        """

        self._reset()

        for path in video_paths:
            video = Video(video_path=path)
            video.map(self.__process_roi)

        return self.__find_apex_phase(self.magnitudes)
    
    
    def process_image_list(self, image_list: List[str]):
        """
        Proses list of images yang sudah diurutkan sebagai frame video.

        Args:
            image_list: List of paths ke file gambar yang akan diproses sebagai frame video.

        Returns:
            Tuple (apex_indices, phases) hasil deteksi apex phase pada sequence gambar.
        """

        self._reset()

        for i in range(len(image_list) - 1):
            prev_frame = cv2.imread(image_list[i])
            curr_frame = cv2.imread(image_list[i + 1])
            self.__process_roi(prev_frame, curr_frame, i)

        return self.__find_apex_phase(self.magnitudes)


    def _reset(self) -> None:
        """Reset state sebelum memproses video baru."""
        self.magnitudes.clear()
        self.horizontal_magnitudes.clear()
        self.vertical_magnitudes.clear()
        self._cached_landmarks = None


    def __detect_landmarks(self, prev_frame: np.ndarray, curr_frame: np.ndarray):
        """
        Deteksi landmark dengan memanfaatkan cache.
        Setiap frame hanya dideteksi satu kali meskipun muncul sebagai
        prev_frame pada iterasi berikutnya.
        """
        prev_landmarks = (
            self._cached_landmarks
            if self._cached_landmarks is not None
            else self.landmarker.detect(prev_frame)
        )

        curr_landmarks = self.landmarker.detect(curr_frame)
        
        self._cached_landmarks = curr_landmarks

        return prev_landmarks, curr_landmarks


    def __extract_rois(self,
                       prev_frame: np.ndarray,
                       curr_frame: np.ndarray,
                       prev_landmarks,
                       curr_landmarks):
        """
        Crop semua ROI dari kedua frame.

        Returns:
            Tuple of lists: (roi_prev_list, roi_next_list, mask_prev_list, mask_next_list)
            Hanya ROI yang berhasil diekstrak dari kedua frame yang dimasukkan.
        """
        roi_prev_list, roi_next_list = [], []
        mask_prev_list, mask_next_list = [], []

        for roi_points, roi_size in self.roi_defs:

            roi_prev, mask_prev = self.landmarker.crop_roi(image=prev_frame,
                                                           landmark_result=prev_landmarks,
                                                           roi_points=roi_points,
                                                           margin=self.margin,
                                                           target_size=roi_size)

            roi_next, mask_next = self.landmarker.crop_roi(image=curr_frame,
                                                           landmark_result=curr_landmarks,
                                                           roi_points=roi_points,
                                                           margin=self.margin,
                                                           target_size=roi_size)

            if roi_prev is None or roi_next is None:
                continue

            roi_prev_list.append(roi_prev)
            roi_next_list.append(roi_next)
            mask_prev_list.append(mask_prev)
            mask_next_list.append(mask_next)

        return roi_prev_list, roi_next_list, mask_prev_list, mask_next_list


    def __process_roi(self,
                      prev_frame: np.ndarray,
                      curr_frame: np.ndarray,
                      frame_index: int) -> None:

        prev_landmarks, curr_landmarks = self.__detect_landmarks(prev_frame, curr_frame)

        roi_prev_list, roi_next_list, mask_prev_list, mask_next_list = self.__extract_rois(prev_frame, 
                                                                                           curr_frame, 
                                                                                           prev_landmarks, 
                                                                                           curr_landmarks)

        if not roi_prev_list:
            self.magnitudes.append(0.0)
            self.horizontal_magnitudes.append(0.0)
            self.vertical_magnitudes.append(0.0)
            return

        if self.mode == "batch":
            frame_magnitude, h_mag, v_mag = self.__compute_batch(roi_prev_list, 
                                                                  roi_next_list, 
                                                                  mask_prev_list, 
                                                                  mask_next_list)
        else:
            frame_magnitude, h_mag, v_mag = self.__compute_single(roi_prev_list, 
                                                                    roi_next_list, 
                                                                    mask_prev_list, 
                                                                    mask_next_list)

        self.magnitudes.append(frame_magnitude)
        self.horizontal_magnitudes.append(h_mag)
        self.vertical_magnitudes.append(v_mag)


    def __compute_single(self,
                         roi_prev_list: list,
                         roi_next_list: list,
                         mask_prev_list: list,
                         mask_next_list: list):
        """
        Hitung optical flow per ROI secara independen, lalu rata-ratakan
        magnitude antar semua ROI menjadi satu nilai frame magnitude.

        Args:
            roi_prev_list: List of cropped ROI dari frame sebelumnya.
            roi_next_list: List of cropped ROI dari frame saat ini.
            mask_prev_list: List of mask valid untuk ROI sebelumnya.
            mask_next_list: List of mask valid untuk ROI saat ini.

        Returns:
            Tuple[float, float, float]: (magnitude, horizontal_magnitude, vertical_magnitude)
                Rata-rata magnitude, horizontal, dan vertical optical flow
                antar semua ROI untuk frame ini.
        """
        roi_magnitudes = []
        roi_horizontal = []
        roi_vertical = []

        for roi_prev, roi_next, mask_prev, mask_next in zip(roi_prev_list, 
                                                            roi_next_list, 
                                                            mask_prev_list, 
                                                            mask_next_list):

            flow = self.tvl1.compute(roi_prev, roi_next)

            dx = flow[..., 0]
            dy = flow[..., 1]

            magnitude = np.hypot(dx, dy)

            valid = (mask_prev > 0) & (mask_next > 0)
            if np.any(valid):
                roi_mean = float(np.mean(magnitude[valid]))
                h_mean = float(np.mean(np.abs(dx[valid])))
                v_mean = float(np.mean(np.abs(dy[valid])))
            else:
                roi_mean = float(np.mean(magnitude))
                h_mean = float(np.mean(np.abs(dx)))
                v_mean = float(np.mean(np.abs(dy)))

            roi_magnitudes.append(roi_mean)
            roi_horizontal.append(h_mean)
            roi_vertical.append(v_mean)

        return (float(np.mean(roi_magnitudes)),
                float(np.mean(roi_horizontal)),
                float(np.mean(roi_vertical)))


    def __compute_batch(self,
                        roi_prev_list: list,
                        roi_next_list: list,
                        mask_prev_list: list,
                        mask_next_list: list):
        """
        Hitung optical flow untuk semua ROI dalam satu pemanggilan TVL1.compute().

        Langkah-langkah:
          1. Pad semua ROI agar memiliki lebar yang seragam (max_w).
          2. Stack secara vertikal menjadi satu gambar batch.
          3. Hitung optical flow sekali pada batch tersebut.
          4. Pisahkan kembali magnitude per ROI menggunakan y_offset.
          5. Hitung mean magnitude, horizontal, dan vertical per ROI berdasarkan mask valid.
          6. Rata-ratakan semua ROI menjadi satu frame magnitude.

        Returns:
            Tuple[float, float, float]: (magnitude, horizontal_magnitude, vertical_magnitude)
        """

        max_w = max(roi.shape[1] for roi in roi_prev_list)

        padded_prev_list, padded_next_list = [], []
        padded_mask_prev_list, padded_mask_next_list = [], []

        for roi_prev, roi_next, mask_prev, mask_next in zip(
            roi_prev_list, roi_next_list, mask_prev_list, mask_next_list
        ):
            pad_w = max_w - roi_prev.shape[1]

            if pad_w > 0:
                roi_prev  = np.pad(roi_prev,  ((0, 0), (0, pad_w), (0, 0)), mode="constant")
                roi_next  = np.pad(roi_next,  ((0, 0), (0, pad_w), (0, 0)), mode="constant")
                mask_prev = np.pad(mask_prev, ((0, 0), (0, pad_w)),         mode="constant")
                mask_next = np.pad(mask_next, ((0, 0), (0, pad_w)),         mode="constant")

            padded_prev_list.append(roi_prev)
            padded_next_list.append(roi_next)
            padded_mask_prev_list.append(mask_prev)
            padded_mask_next_list.append(mask_next)

        batch_prev = np.vstack(padded_prev_list)
        batch_next = np.vstack(padded_next_list)

        flow = self.tvl1.compute(batch_prev, batch_next)
        dx_batch = flow[..., 0]
        dy_batch = flow[..., 1]
        magnitude_batch = np.hypot(dx_batch, dy_batch)

        roi_magnitudes = []
        roi_horizontal = []
        roi_vertical = []
        y_offset = 0

        for roi_prev, mask_prev, mask_next in zip(padded_prev_list, 
                                                  padded_mask_prev_list, 
                                                  padded_mask_next_list):
            
            h = roi_prev.shape[0]
            magnitude = magnitude_batch[y_offset:y_offset + h, :]
            dx = dx_batch[y_offset:y_offset + h, :]
            dy = dy_batch[y_offset:y_offset + h, :]

            valid = (mask_prev > 0) & (mask_next > 0)
            if np.any(valid):
                roi_mean = float(np.mean(magnitude[valid]))
                h_mean = float(np.mean(np.abs(dx[valid])))
                v_mean = float(np.mean(np.abs(dy[valid])))
            else:
                roi_mean = float(np.mean(magnitude))
                h_mean = float(np.mean(np.abs(dx)))
                v_mean = float(np.mean(np.abs(dy)))

            roi_magnitudes.append(roi_mean)
            roi_horizontal.append(h_mean)
            roi_vertical.append(v_mean)

            y_offset += h

        return (float(np.mean(roi_magnitudes)),
                float(np.mean(roi_horizontal)),
                float(np.mean(roi_vertical)))


    def __find_apex_phase(self, magnitudes: List[float]):

        smoothed = ApexSmoother.smooth(signal=magnitudes)
        self.magnitudes = smoothed

        smoothed_horizontal = ApexSmoother.smooth(signal=self.horizontal_magnitudes)
        self.horizontal_magnitudes = smoothed_horizontal

        smoothed_vertical = ApexSmoother.smooth(signal=self.vertical_magnitudes)
        self.vertical_magnitudes = smoothed_vertical

        apex_indices = self.apex_phase.find_apex(signal=smoothed)

        phases = self.apex_phase.find_phase(signal=smoothed,
                                            apex_indices=apex_indices,
                                            cutoff_ratio=0)

        return apex_indices, phases