import numpy as np
import cv2

SMOOTHING_RADIUS = 50

def moving_average(curve, radius):
    window_size = 2 * radius + 1
    filter_window = np.ones(window_size) / window_size
    curve_padded = np.pad(curve, (radius, radius), mode='edge')
    smoothed_curve = np.convolve(curve_padded, filter_window, mode='same')
    return smoothed_curve[radius:-radius]

def smooth_trajectory(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

def fix_border(frame):
    height, width = frame.shape[:2]
    transformation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 1.04)
    return cv2.warpAffine(frame, transformation_matrix, (width, height))

# Input video
cap = cv2.VideoCapture('buoy_video.mp4')

n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_out.mp4', fourcc, fps, (width, height))

# Read first frame
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

transforms = np.zeros((n_frames - 1, 3), np.float32)

for i in range(n_frames - 1):
    prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    ret, curr_frame = cap.read()
    
    if not ret:
        break
    
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None)
    
    valid_idx = np.where(status == 1)[0]
    prev_points = prev_points[valid_idx]
    curr_points = curr_points[valid_idx]
    
    assert prev_points.shape == curr_points.shape
    
    transform_matrix = cv2.estimateRigidTransform(prev_points, curr_points, fullAffine=False)
    
    dx = transform_matrix[0, 2]
    dy = transform_matrix[1, 2]
    da = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
    
    transforms[i] = [dx, dy, da]
    prev_gray = curr_gray

trajectory = np.cumsum(transforms, axis=0)
smoothed_trajectory = smooth_trajectory(trajectory)
difference = smoothed_trajectory - trajectory
transforms_smooth = transforms + difference

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for i in range(n_frames - 1):
    ret, frame = cap.read()
    if not ret:
        break

    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    transform_matrix = np.array([
        [np.cos(da), -np.sin(da), dx],
        [np.sin(da),  np.cos(da), dy]
    ], dtype=np.float32)

    stabilized_frame = cv2.warpAffine(frame, transform_matrix, (width, height))
    stabilized_frame = fix_border(stabilized_frame)

    frame_out = cv2.hconcat([frame, stabilized_frame])
    
    if frame_out.shape[1] > 1920:
        frame_out = cv2.resize(frame_out, (frame_out.shape[1] // 2, frame_out.shape[0] // 2))
    
    cv2.imshow("Before and After", frame_out)
    cv2.waitKey(10)

    out.write(stabilized_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
