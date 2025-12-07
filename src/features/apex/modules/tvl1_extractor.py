import cv2
import numpy as np
import matplotlib.pyplot as plt


class TVL1:
    def __init__(self):
        """
        Initialize the TVL1 optical flow extractor
        Uses GPU acceleration if available.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """
    
        # Check if CUDA available on the system
        # If available, use the GPU version for TVL1 optical flow processing
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            
            # If the CUDA support for TVL1 then initialize model with it
            if hasattr(cv2.cuda, "OpticalFlowDual_TVL1_create"):
                self.tvl1           = cv2.cuda.OpticalFlowDual_TVL1_create()
                self.gpumat_prev    = cv2.cuda_GpuMat()
                self.gpumat_next    = cv2.cuda_GpuMat()
                self.gpumat_flow    = cv2.cuda_GpuMat()

            # Otherwise, fallback to CPU version
        else:
            self.tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()


        # Setting up parameters for optimizing TVL1 performance
        self.tvl1.setLambda(0.15)
        self.tvl1.setTheta(0.3)
        self.tvl1.setTau(0.25)

        self.flow   = None
        self.flows  = []

        # Reset for each new instance
        # This ensures no residual data from previous computations
        self.reset()


    def compute(self, prev_image, next_image):
        """
        Compute the TVL1 optical flow between two images.

        Args:
            prev_image (numpy.ndarray): The previous image frame.
            next_image (numpy.ndarray): The next image frame.

        Returns:
            numpy.ndarray: The computed optical flow.

        Raises:
            None
        """
        
        gray_prev = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
        gray_next = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)

        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0 and hasattr(cv2.cuda, "OpticalFlowDual_TVL1_create"):
            self.gpumat_prev.upload(gray_prev)
            self.gpumat_next.upload(gray_next)

            flow_gpu = self.tvl1.calc(self.gpumat_prev, self.gpumat_next, self.gpumat_flow)

            flow = flow_gpu.download()
        else:
            flow = self.tvl1.calc(gray_prev, gray_next, None)

        self.flow = flow

        return self
    

    def compute_sequence(self, frames):
        """
        Compute the TVL1 optical flow for a sequence of video frames.

        Args:
            frames (list of numpy.ndarray): List of video frames.
        
        Returns:
            list of numpy.ndarray: List of computed optical flows between consecutive frames.
        
        Raises:
            None
        """

        self.flows = []

        for i in range(len(frames) - 1):
            prev_frame = frames[i]
            next_frame = frames[i + 1]

            flow = self.compute(prev_frame, next_frame).get_flow()

            self.flows.append(flow)

        return self.flows


    def magnitude(self):
        """
        Calculate the magnitude of the optical flow vectors.
        
        Returns:
            numpy.ndarray: The magnitude of the optical flow.

        Raises:
            ValueError: If optical flow has not been computed yet.
        """

        if self.flow is None:
            raise ValueError("Optical flow has not been computed yet.")
        
        flow_x = self.flow[:, :, 0]
        flow_y = self.flow[:, :, 1]

        magnitude = np.sqrt(flow_x**2 + flow_y**2)

        return magnitude
    

    def angle(self):
        """
        Calculate the angle of the optical flow vectors.
        
        Returns:
            numpy.ndarray: The angle of the optical flow in radians.

        Raises:
            ValueError: If optical flow has not been computed yet.
        """

        if self.flow is None:
            raise ValueError("Optical flow has not been computed yet.")
        
        flow_x = self.flow[:, :, 0]
        flow_y = self.flow[:, :, 1]

        angle = np.arctan2(flow_y, flow_x)

        return angle
    

    def get_flow(self):
        """
        Get the computed optical flow.

        Returns:
            numpy.ndarray: The computed optical flow.

        Raises:
            ValueError: If optical flow has not been computed yet.
        """

        if self.flow is None:
            raise ValueError("Optical flow has not been computed yet.")
        
        return self.flow
    

    def reset(self):
        """
        Reset the internal state of the TVL1 extractor.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        self.flow = None


    def plot_quiver(self, step=16):
        """
        Plot the optical flow using quiver plot.

        Args:
            step (int): The step size for downsampling the flow vectors for visualization.

        Returns:
            None
        
        Raises:
            ValueError: If optical flow has not been computed yet.
        """

        if self.flow is None:
            raise ValueError("Optical flow has not been computed yet.")
        
        h, w = self.flow.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)

        fx = self.flow[y, x, 0]
        fy = self.flow[y, x, 1]

        plt.figure(figsize=(10, 10))
        plt.quiver(x, y, fx, fy, color='r', angles='xy', scale_units='xy', scale=1)
        plt.gca().invert_yaxis()
        plt.title('Optical Flow Quiver Plot')
        plt.show()