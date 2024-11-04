import cv2
import numpy as np

class MotionModel:
    """Class to handle motion model, motion compensation, and background subtraction."""

    @staticmethod
    def apply_motion_model(prev_gray_frame, curr_gray_frame):
        """Estimate motion between two grayscale frames using optical flow."""
        flow = cv2.calcOpticalFlowFarneback(prev_gray_frame, curr_gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        dx, dy = np.median(flow[..., 0]), np.median(flow[..., 1])
        return dx, dy

    @staticmethod
    def compensate_motion(frame, dx, dy):
        """Apply motion compensation to a frame."""
        return cv2.warpAffine(frame, np.float32([[1, 0, -dx], [0, 1, -dy]]), (frame.shape[1], frame.shape[0]))

    @staticmethod
    def apply_background_subtraction(frame, background):
        """Apply background subtraction and preprocess to enhance tracking."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(background_gray, frame_gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        # thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # # Apply morphological operations to reduce noise
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return thresh
