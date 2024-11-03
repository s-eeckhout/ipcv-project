import cv2
import numpy as np
import depthCalculation2
import importlib
from motionModel import MotionModel

# Reload the depth calculation module if needed
importlib.reload(depthCalculation2)


class BuoyTracker:
    """Class to track a buoy using the CSRT tracker and motion compensation."""

    def __init__(self, video_path, initial_bbox, frame_number):
        self.video_path = video_path
        self.frame_number = frame_number
        self.tracker = self.set_csrt_params()
        self.cap = cv2.VideoCapture(video_path)
        self.frame_count = frame_number
        self.background = self.initialize_background()
        self.prev_gray_frame = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)

        # initialize ROI, if preferred to select the ROI manually; pass None as initial_bbox
        if initial_bbox is not None: self.initial_bbox = initial_bbox
        else: self.initial_bbox = cv2.selectROI("Select ROI", self.background, fromCenter=False, showCrosshair=True)
        
        self.tracker.init(self.background, self.initial_bbox)

    @staticmethod
    def set_csrt_params():
        """Set and return custom CSRT tracker parameters."""
        default_params = {
            'padding': 3.0, 'template_size': 200.0, 'gsl_sigma': 1.0,
            'hog_orientations': 9.0, 'num_hog_channels_used': 18, 'hog_clip': 0.2,
            'use_hog': 1, 'use_color_names': 1, 'use_gray': 1, 'use_rgb': 0,
            'window_function': 'hann', 'kaiser_alpha': 3.75, 'cheb_attenuation': 45.0,
            'filter_lr': 0.02, 'admm_iterations': 4, 'number_of_scales': 100,
            'scale_sigma_factor': 0.25, 'scale_model_max_area': 512.0, 'scale_lr': 0.025,
            'scale_step': 1.02, 'use_channel_weights': 1, 'weights_lr': 0.02,
            'use_segmentation': 1, 'histogram_bins': 16, 'background_ratio': 2,
            'histogram_lr': 0.04, 'psr_threshold': 0.035
        }
        params = {
            'number_of_scales': 33, 'scale_step': 1.05, 'padding': 1.5,
            'psr_threshold': 0.05, 'use_channel_weights': 0
        }
        final_params = {**default_params, **params}
        
        param_handler = cv2.TrackerCSRT_Params()
        for key, val in final_params.items():
            setattr(param_handler, key, val)
        return cv2.TrackerCSRT_create(param_handler)
    
    def select_initial_position(video_path):
        """
        Plays the first few frames of the video slowly. When the user presses 's',
        the video pauses, and they can select the initial position of the buoy.
        Returns the (x, y) coordinates of the selected point and the frame number.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return None, None

        initial_position = None
        frame_number = 0

        # Callback function to capture mouse click
        def mouse_callback(event, x, y, flags, param):
            nonlocal initial_position
            if event == cv2.EVENT_LBUTTONDOWN:
                initial_position = (x, y)
                print(f"Initial position selected at: {initial_position} in frame {frame_number}")
                cv2.destroyWindow("Select Initial Position")  # Close the window after selection

        # Set up window and callback
        cv2.namedWindow("Select Initial Position")
        cv2.setMouseCallback("Select Initial Position", mouse_callback)

        frame_delay = 500  # Delay in milliseconds to slow down frames

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Reached the end of the video or encountered an error.")
                break

            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Current frame number
            cv2.imshow("Select Initial Position", frame)
            key = cv2.waitKey(frame_delay) & 0xFF

            if key == ord('s'):  # Press 's' to select initial position
                print("Press 's' detected. Click on the frame to select the initial position.")
                while initial_position is None:
                    cv2.waitKey(1)
                break
            elif key == ord('q'):  # Press 'q' to quit
                print("Selection canceled.")
                break

        cap.release()
        return initial_position, frame_number

    def initialize_background(self):
        """Initialize and return the background frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        success, background = self.cap.read()
        if not success:
            print("Failed to read the background frame.")
            self.cap.release()
            exit()
        return background

    def draw_annotations(self, frame, bbox, distance, frame_count):
        """Draw the bounding box, distance text, and frame count on the frame."""
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 65, 55), 2)

        x_coor = x + w // 2
        y_coor = y + h // 2
        text = f"Distance {round(distance, 4)}m"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x, text_y = x, y - 10
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), 
                      (text_x + text_size[0], text_y + 5), (200, 200, 200), -1)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        cv2.putText(frame, f"Frame {frame_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)

    def run(self):
        """Main function to run the buoy tracking."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1
            success, bbox = self.tracker.update(frame)
            curr_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            dx, dy = MotionModel.apply_motion_model(self.prev_gray_frame, curr_gray_frame)
            motion_compensated_frame = MotionModel.compensate_motion(frame, dx, dy)
            motion_free_frame = MotionModel.apply_background_subtraction(motion_compensated_frame, self.background)

            success, bbox = self.tracker.update(motion_compensated_frame)
            if success:
                distance_buoy = depthCalculation2.detect_horizontal_lines_in_video(frame, bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2)
                self.draw_annotations(frame, bbox, distance_buoy, self.frame_count)
            else:
                cv2.putText(frame, "Buoy Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            self.prev_gray_frame = curr_gray_frame
            cv2.imshow("Motion Free Frame", motion_free_frame)
            cv2.imshow("Buoy Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    initial_bbox = (593, 476, 37, 17)
    frame_number = 23
    tracker = BuoyTracker("stabilized_video.mp4", initial_bbox, frame_number)
    tracker.run()