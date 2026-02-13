import numpy as np
from typing import List, Tuple
from src.video.modules import Video
from scipy.signal import savgol_filter
from src.face.modules import FaceRoiPoints, FaceRoiSizes
from src.face.modules import FaceLandmark
from src.optical_flow.modules import TVL1
from src.apex.modules import ApexPhase, ApexSmoother


class ApexPhaseSpotter:

    def __init__(self, distance_threshold: int = 11, prominence_threshold: float = 0.3):
        """
        Inisialisasi instance untuk deteksi fase apex pada sebuah video

        Args:
            distance_threshold (int): Jarak minimum antara puncak apex yang terdeteksi.
            prominence_threshold (float): Ambang batas prominensi untuk mendeteksi puncak apex.
        """
    
        self.landmarker = FaceLandmark()

        self.tvl1 = TVL1()

        self.apex_phase = ApexPhase(distance_threshold=distance_threshold,
                                    prominence_threshold=prominence_threshold)
        
        self.magnitudes = []

    
    def process(self, video_path: str) -> Tuple[List[int], List[int]]:
        """
        Memproses video untuk mendeteksi indeks apex dan fase

        Args:
            video_path (str): Path ke file video yang akan diproses

        Returns:
            Tuple[List[int], List[int]]: Indeks apex dan fase yang terdeteksi
        """

        video = Video(video_path=video_path)

        video.map(self.__process_with_roi)

        apex_indices, phases = self.__find_apex_phase(self.magnitudes)

        return apex_indices, phases

    
    def __process_with_roi(self, prev_frame: np.ndarray, curr_frame: np.ndarray, frame_index: int) -> np.ndarray:

        prev_landmarks = self.landmarker.detect(prev_frame)
        curr_landmarks = self.landmarker.detect(curr_frame)

        cropped_left_eye_prev, left_eye_roi_mask               = self.landmarker.crop_roi(image=prev_frame.copy(),
                                                                                          landmark_result=prev_landmarks,
                                                                                          roi_points=FaceRoiPoints.LEFT_EYE_POINTS,
                                                                                          target_size=FaceRoiSizes.EYE_SIZE)
        
        cropped_left_eye_next, left_eye_roi_mask               = self.landmarker.crop_roi(image=curr_frame.copy(),
                                                                                          landmark_result=curr_landmarks,
                                                                                          roi_points=FaceRoiPoints.LEFT_EYE_POINTS,
                                                                                          target_size=FaceRoiSizes.EYE_SIZE)
        
        cropped_right_eye_prev, right_eye_roi_mask             = self.landmarker.crop_roi(image=prev_frame.copy(),
                                                                                          landmark_result=prev_landmarks,
                                                                                          roi_points=FaceRoiPoints.RIGHT_EYE_POINTS,
                                                                                          target_size=FaceRoiSizes.EYE_SIZE)
        
        cropped_right_eye_next, right_eye_roi_mask             = self.landmarker.crop_roi(image=curr_frame.copy(),
                                                                                          landmark_result=curr_landmarks,
                                                                                          roi_points=FaceRoiPoints.RIGHT_EYE_POINTS,
                                                                                          target_size=FaceRoiSizes.EYE_SIZE)
        
        cropped_left_eyebrow_prev, left_eyebrow_roi_mask       = self.landmarker.crop_roi(image=prev_frame.copy(),
                                                                                          landmark_result=prev_landmarks,
                                                                                          roi_points=FaceRoiPoints.LEFT_EYEBROW_POINTS,
                                                                                          target_size=FaceRoiSizes.EYEBROW_SIZE)
        
        cropped_left_eyebrow_next, left_eyebrow_roi_mask       = self.landmarker.crop_roi(image=curr_frame.copy(),
                                                                                          landmark_result=curr_landmarks,
                                                                                          roi_points=FaceRoiPoints.LEFT_EYEBROW_POINTS,
                                                                                          target_size=FaceRoiSizes.EYEBROW_SIZE)
        
        cropped_right_eyebrow_prev, right_eyebrow_roi_mask     = self.landmarker.crop_roi(image=prev_frame.copy(),
                                                                                          landmark_result=prev_landmarks,
                                                                                          roi_points=FaceRoiPoints.RIGHT_EYEBROW_POINTS,
                                                                                          target_size=FaceRoiSizes.EYEBROW_SIZE)
        
        cropped_right_eyebrow_next, right_eyebrow_roi_mask     = self.landmarker.crop_roi(image=curr_frame.copy(),
                                                                                          landmark_result=curr_landmarks,
                                                                                          roi_points=FaceRoiPoints.RIGHT_EYEBROW_POINTS,
                                                                                          target_size=FaceRoiSizes.EYEBROW_SIZE)
        
        cropped_lips_prev, lips_roi_mask                       = self.landmarker.crop_roi(image=prev_frame.copy(),
                                                                                          landmark_result=prev_landmarks,
                                                                                          roi_points=FaceRoiPoints.LIPS_POINTS,
                                                                                          target_size=FaceRoiSizes.LIPS_SIZE)
        
        cropped_lips_next, lips_roi_mask                       = self.landmarker.crop_roi(image=curr_frame.copy(),
                                                                                          landmark_result=curr_landmarks,
                                                                                          roi_points=FaceRoiPoints.LIPS_POINTS,
                                                                                          target_size=FaceRoiSizes.LIPS_SIZE)
        
        flow_left_eye           = self.tvl1.compute(cropped_left_eye_prev, cropped_left_eye_next)
        flow_right_eye          = self.tvl1.compute(cropped_right_eye_prev, cropped_right_eye_next)
        flow_left_eyebrow       = self.tvl1.compute(cropped_left_eyebrow_prev, cropped_left_eyebrow_next)
        flow_right_eyebrow      = self.tvl1.compute(cropped_right_eyebrow_prev, cropped_right_eyebrow_next)
        flow_lips               = self.tvl1.compute(cropped_lips_prev, cropped_lips_next)

        flow_left_eye[..., 0]    *= (left_eye_roi_mask > 0)
        flow_left_eye[..., 1]    *= (left_eye_roi_mask > 0)

        flow_right_eye[..., 0]   *= (right_eye_roi_mask > 0)
        flow_right_eye[..., 1]   *= (right_eye_roi_mask > 0)

        flow_left_eyebrow[..., 0]    *= (left_eyebrow_roi_mask > 0)
        flow_left_eyebrow[..., 1]    *= (left_eyebrow_roi_mask > 0)

        flow_right_eyebrow[..., 0]   *= (right_eyebrow_roi_mask > 0)
        flow_right_eyebrow[..., 1]   *= (right_eyebrow_roi_mask > 0)

        flow_lips[..., 0]    *= (lips_roi_mask > 0)
        flow_lips[..., 1]    *= (lips_roi_mask > 0)

        magnitude_left_eye          = np.sqrt(flow_left_eye[..., 0]**2 + flow_left_eye[..., 1]**2)
        magnitude_right_eye         = np.sqrt(flow_right_eye[..., 0]**2 + flow_right_eye[..., 1]**2)
        magnitude_left_eyebrow      = np.sqrt(flow_left_eyebrow[..., 0]**2 + flow_left_eyebrow[..., 1]**2)
        magnitude_right_eyebrow     = np.sqrt(flow_right_eyebrow[..., 0]**2 + flow_right_eyebrow[..., 1]**2)
        magnitude_lips              = np.sqrt(flow_lips[..., 0]**2 + flow_lips[..., 1]**2)

        mean_magnitude_left_eye        = np.mean(magnitude_left_eye)
        mean_magnitude_right_eye       = np.mean(magnitude_right_eye)
        mean_magnitude_left_eyebrow    = np.mean(magnitude_left_eyebrow)
        mean_magnitude_right_eyebrow   = np.mean(magnitude_right_eyebrow)
        mean_magnitude_lips            = np.mean(magnitude_lips)

        mean_magnitude = np.mean([
            mean_magnitude_left_eye,
            mean_magnitude_right_eye,
            mean_magnitude_left_eyebrow,
            mean_magnitude_right_eyebrow,
            mean_magnitude_lips
        ])

        self.magnitudes.append(mean_magnitude)


    def __find_apex_phase(self, magnitudes: List[np.ndarray]) -> Tuple[List[int], List[int]]:
        """
        Mendeteksi indeks apex dan fase pada sinyal magnitudo

        Args:
            magnitudes (List[np.ndarray]): Daftar magnitudo yang telah dihaluskan

        Returns:
            Tuple[List[int], List[int]]: Indeks apex dan fase yang terdeteksi
        """
        window_length = ApexSmoother.calculate_window_length(len(magnitudes))

        polyorder = ApexSmoother.calculate_polyorder(window_length)

        smoothed_magnitudes = savgol_filter(magnitudes, window_length, polyorder)

        apex_indices = self.apex_phase.find_apex(signal=smoothed_magnitudes)

        phases = self.apex_phase.find_phase(signal=smoothed_magnitudes, apex_indices=apex_indices)

        return apex_indices, phases