import cv2
import numpy as np
from typing import List, Literal, Tuple, Optional

from src.video.modules import Video
from src.face.modules import FaceLandmark, FaceRoiPoints, FaceRoiSizes
from src.optical_flow.modules import TVL1
from src.apex.modules.v3 import ApexSmoother
from src.apex.modules.v3.apex_wave_detector import ApexWaveDetector
from src.apex.modules.v3.apex_wave_validator import ApexWaveValidator


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
        self.wave_detector = ApexWaveDetector()
        self.wave_validator = ApexWaveValidator()

        self.roi_defs = [
            (FaceRoiPoints.LEFT_EYE_POINTS,    FaceRoiSizes.EYE_SIZE),
            (FaceRoiPoints.RIGHT_EYE_POINTS,   FaceRoiSizes.EYE_SIZE),
            (FaceRoiPoints.LIPS_POINTS,        FaceRoiSizes.LIPS_SIZE),
            (FaceRoiPoints.LEFT_EYEBROW_POINTS,  FaceRoiSizes.EYEBROW_SIZE),
            (FaceRoiPoints.RIGHT_EYEBROW_POINTS, FaceRoiSizes.EYEBROW_SIZE),
        ]

        self.magnitudes: List[float] = []

        # Cache landmark frame sebelumnya agar tidak ada deteksi ganda.
        # curr_landmarks iterasi N menjadi prev_landmarks iterasi N+1.
        self._cached_landmarks = None


    def process(self, video_path: str, precomputed_magnitudes: Optional[List[float]] = None):
        """
        Proses satu video dan kembalikan apex indices beserta phases.
        
        Args:
            video_path: Path ke file video yang akan diproses.
            precomputed_magnitudes: List magnitude yang sudah dihitung sebelumnya (opsonal).

        Returns:
            Tuple (apex_indices, phases) hasil deteksi apex phase pada video.
        """

        self._reset()

        if precomputed_magnitudes is not None:
            self.magnitudes = list(precomputed_magnitudes)
        else:
            video = Video(video_path=video_path)
            video.map(self.__process_roi)

        return self.__find_apex_phase(self.magnitudes)


    def process_videos(self, video_paths: List[str], precomputed_magnitudes: Optional[List[float]] = None):
        """
        Proses beberapa video secara berurutan dan kembalikan apex indices beserta phases.
        
        Args:
            video_paths: List of paths ke file video yang akan diproses.
            precomputed_magnitudes: List magnitude yang sudah dihitung sebelumnya (opsonal).

        Returns:
            Tuple (apex_indices, phases) hasil deteksi apex phase pada semua video.
        """

        self._reset()

        if precomputed_magnitudes is not None:
            self.magnitudes = list(precomputed_magnitudes)
        else:
            for path in video_paths:
                video = Video(video_path=path)
                video.map(self.__process_roi)

        return self.__find_apex_phase(self.magnitudes)

    
    def process_images(self, images: List[str]) -> Tuple[List[int], List[int]]:
        """
        Proses beberapa gambar secara berurutan dan kembalikan apex indices beserta phases.
        
        Args:
            images: List of paths ke file gambar yang akan diproses.

        Returns:
            Tuple (apex_indices, phases) hasil deteksi apex phase pada semua gambar.
        """
        self._reset()

        if len(images) < 2:
            return [], []

        prev_frame = cv2.imread(images[0])
        if prev_frame is None:
            raise ValueError(f"Could not read image at {images[0]}")

        for frame_index in range(1, len(images)):
            curr_frame = cv2.imread(images[frame_index])
            if curr_frame is None:
                raise ValueError(f"Could not read image at {images[frame_index]}")

            self.__process_roi(prev_frame, curr_frame, frame_index - 1)

            prev_frame = curr_frame

        return self.__find_apex_phase(self.magnitudes)


    def _reset(self) -> None:
        """Reset state sebelum memproses video baru."""
        self.magnitudes.clear()
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
            return

        if self.mode == "batch":
            frame_magnitude = self.__compute_batch(roi_prev_list, 
                                                   roi_next_list, 
                                                   mask_prev_list, 
                                                   mask_next_list)
        else:
            frame_magnitude = self.__compute_single(roi_prev_list, 
                                                    roi_next_list, 
                                                    mask_prev_list, 
                                                    mask_next_list)

        self.magnitudes.append(frame_magnitude)


    def __compute_single(self,
                         roi_prev_list: list,
                         roi_next_list: list,
                         mask_prev_list: list,
                         mask_next_list: list) -> float:
        """
        Hitung optical flow per ROI secara independen, lalu rata-ratakan
        magnitude antar semua ROI menjadi satu nilai frame magnitude.

        Args:
            roi_prev_list: List of cropped ROI dari frame sebelumnya.
            roi_next_list: List of cropped ROI dari frame saat ini.
            mask_prev_list: List of mask valid untuk ROI sebelumnya.
            mask_next_list: List of mask valid untuk ROI saat ini.

        Returns:
            float: Rata-rata magnitude optical flow antar semua ROI untuk frame ini.
        """
        roi_magnitudes = []

        for roi_prev, roi_next, mask_prev, mask_next in zip(roi_prev_list, 
                                                            roi_next_list, 
                                                            mask_prev_list, 
                                                            mask_next_list):

            flow = self.tvl1.compute(roi_prev, roi_next)
            magnitude = np.hypot(flow[..., 0], flow[..., 1])

            valid = (mask_prev > 0) & (mask_next > 0)
            roi_mean = float(np.mean(magnitude[valid]) if np.any(valid) else np.mean(magnitude))
            roi_magnitudes.append(roi_mean)

        return float(np.mean(roi_magnitudes))


    def __compute_batch(self,
                        roi_prev_list: list,
                        roi_next_list: list,
                        mask_prev_list: list,
                        mask_next_list: list) -> float:
        """
        Hitung optical flow untuk semua ROI dalam satu pemanggilan TVL1.compute().

        Langkah-langkah:
          1. Pad semua ROI agar memiliki lebar yang seragam (max_w).
          2. Stack secara vertikal menjadi satu gambar batch.
          3. Hitung optical flow sekali pada batch tersebut.
          4. Pisahkan kembali magnitude per ROI menggunakan y_offset.
          5. Hitung mean magnitude per ROI berdasarkan mask valid.
          6. Rata-ratakan semua ROI menjadi satu frame magnitude.
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
        magnitude_batch = np.hypot(flow[..., 0], flow[..., 1])

        roi_magnitudes = []
        y_offset = 0

        for roi_prev, mask_prev, mask_next in zip(padded_prev_list, 
                                                  padded_mask_prev_list, 
                                                  padded_mask_next_list):
            
            h = roi_prev.shape[0]
            magnitude = magnitude_batch[y_offset:y_offset + h, :]

            valid = (mask_prev > 0) & (mask_next > 0)
            roi_mean = float(np.mean(magnitude[valid]) if np.any(valid) else np.mean(magnitude))
            roi_magnitudes.append(roi_mean)

            y_offset += h

        return float(np.mean(roi_magnitudes))


    def __find_apex_phase(self, magnitudes: List[float]):

        smoothed = ApexSmoother.smooth(signal=magnitudes)
        self.magnitudes = smoothed

        # Wave-based detection: segmentasi → validasi → extract apex & phases
        waves = self.wave_detector.detect_waves(signal=smoothed)
        waves = self.wave_validator.validate(waves=waves, signal=smoothed)

        apex_indices = [w["apex"] for w in waves]

        phases = {
            w["apex"]: {
                "start": w["onset"],
                "end": w["offset"]
            }
            for w in waves
        }

        return apex_indices, phases