import argparse
import cv2
import sys
import numpy as np
import cv2.aruco as aruco

markerLength = 0.07  # Define the marker length in meters (adjust as needed)
camMatrix = np.array([[3046, 0, 1542], [0, 3024, 2007], [0, 0, 1]], dtype=np.float32) # Calibrated mobile camera - iphone 14 plus 
distCoeffs = np.zeros((0, 0), dtype=np.float32)

template_UTlogo = 'buoyTemplate.png' 
UTlogo_template = cv2.imread(template_UTlogo, cv2.IMREAD_COLOR)
if UTlogo_template is None:
    print("Error: UT logo Template file not found.")
    sys.exit()

# Function to apply template matching to a frame
def apply_template_matching(frame, template, method=cv2.TM_CCOEFF_NORMED):
    result = cv2.matchTemplate(frame, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h, w = template.shape[:2]
    bottom_right = (top_left[0] + w, top_left[1] + h)
    matched_frame = cv2.rectangle(frame.copy(), top_left, bottom_right, (255,0,0), 2)
    return matched_frame, top_left, bottom_right

# Function to stabilize video using optical flow
def stabilize_video(prev_frame, curr_frame, prev_transform):
    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Detect good features to track in the previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    
    # Calculate optical flow (movement) from previous frame to current frame
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    
    # Filter only valid points
    valid_prev_pts = prev_pts[status == 1]
    valid_curr_pts = curr_pts[status == 1]
    
    # Estimate the transformation matrix (rigid transformation)
    transform = cv2.estimateAffinePartial2D(valid_prev_pts, valid_curr_pts)[0]  # Use only translation, rotation, and scaling
    
    # If transform is None (happens occasionally), use the previous transform
    if transform is None:
        transform = prev_transform
    
    # Apply the transformation to stabilize the current frame
    stabilized_frame = cv2.warpAffine(curr_frame, transform, (curr_frame.shape[1], curr_frame.shape[0]))
    
    return stabilized_frame, transform

def main(input_video_file: str, output_video_file: str) -> None:
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    prev_frame = None  
    prev_transform = np.eye(2, 3, dtype=np.float32)  # Initial transformation matrix (identity)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Check key press for quitting
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break

            # Stabilize the video if there is a previous frame
            if prev_frame is not None:
                stabilized_frame, prev_transform = stabilize_video(prev_frame, frame, prev_transform)
            else:
                stabilized_frame = frame.copy()
            
            prev_frame = frame.copy()  # Update the previous frame
            
           
                # Apply template matching to the stabilized frame
            matched_frame, top_left, bottom_right = apply_template_matching(stabilized_frame, UTlogo_template)
                
                # Add subtitles to indicate template matching
            cv2.putText(matched_frame, "UT logo Template Matching", [50, frame_height - 50], 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 30)
            cv2.putText(matched_frame, f"Match Location: {top_left}", [50, frame_height - 80], 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 30)
                
            out.write(matched_frame)
            # cv2.imshow('Template Matching Frame', matched_frame)

            frame = cv2.putText(stabilized_frame, "Original Frame", [50, frame_height - 80], 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, 30)
            out.write(stabilized_frame)
            cv2.imshow('Original Frame', stabilized_frame)

            # Press 'q' to exit
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything is done, release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()
    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")
    main(args.input, args.output)
