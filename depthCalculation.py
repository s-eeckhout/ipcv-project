import cv2
import math
import numpy as np

kernelErode = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1], 
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]], dtype=np.uint8)

kerneDilate = np.ones((6, 6), dtype=np.uint8)

def calculate_distance(px_obj,py_obj,horizon_ox1,horizon_oy1,horizon_ox2,horizon_oy2,earth_radius, camera_fy, camera_height):
    #line_y = (((horizon_oy2-horizon_oy1)/(horizon_ox2-horizon_ox1))*px_obj + horizon_oy1)
    line_y = (horizon_oy2+horizon_oy1)/2
    pixel_distance= math.sqrt((px_obj - px_obj) ** 2 + (line_y - py_obj) ** 2)
    theta_dy = math.atan(pixel_distance / camera_fy)
    principle_line = math.sqrt(2*camera_height*earth_radius + camera_height*camera_height)
    horizon_dist = math.sqrt(principle_line*principle_line-camera_height*camera_height)
    theta_c = math.asin(camera_height/horizon_dist)
    distance_to_object = camera_height/math.tan(theta_c+theta_dy)

    return distance_to_object

def detect_horizontal_lines_in_video(video_path, output_path):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create VideoWriter object to save the combined frames as MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 output
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height)) #if you change the dimensions you can save the vidoe 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        display_frame1=frame;

        # Resize frame to half its original size
        frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        display_frame1 = cv2.resize(display_frame1, (0, 0), fx=1, fy=1)
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(frame, (15, 15), 0.35)

        # Convert to grayscale
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        #edges = cv2.Canny(gray, 150, 250, apertureSize=3) #for dialated
        edges = cv2.Canny(gray, 140, 320, apertureSize=3)  
        
        binary_image = edges
        # Apply dilation
        dilated_image = cv2.dilate(binary_image, kerneDilate, 1)

        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        # Apply Hough Line Transform to detect horizontal lines
        #lines = cv2.HoughLines(edges, 1, np.pi/180, 90, None, 0, 0)
        #lines = cv2.HoughLines(dilated_image, 1, np.pi/180, 620, None, 0, 0) #best ever
        # lines = cv2.HoughLines(dilated_image, 1, np.pi/180, 700, None, 0, 0) #best ever for MAH01462.mp4
        lines = cv2.HoughLines(dilated_image, 1, np.pi/180, 850, None, 0, 0) #best ever for stabilized
        horizontal_lines = np.zeros_like(frame)

        angle = 0.05
        total_rho = 0
        total_theta = 0
        count = 0
        height, width = frame.shape[:2]
        line_length = width



        # Filter and draw only horizontal lines
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                # Check if theta is between -angle and angle (in radians)
                #if -angle <= np.radians(theta) <= angle:  
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + line_length * (-b)), int(y0 + line_length * (a)))
                pt2 = (int(x0 - line_length * (-b)), int(y0 - line_length * (a)))
                #cv2.line(binary_image, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(display_frame1, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)  # Blue line for average
                # print("Theta in degrees:", np.degrees(theta))  # Print theta in degrees
               # Accumulate rho and theta
                if(abs(90-np.degrees(theta))<3 and 400 < rho < 620):
                    total_rho += rho
                    total_theta += theta
                    count += 1

            # Calculate average line if any lines were found
            if count > 0:
                avg_rho = total_rho / count
                avg_theta = total_theta / count
                
                # Calculate endpoints of the averaged line
                a_avg = math.cos(avg_theta)
                b_avg = math.sin(avg_theta)
                x0_avg = a_avg * avg_rho
                y0_avg = b_avg * avg_rho
                pt1_avg = (int(x0_avg + line_length * (-b_avg)), int(y0_avg + line_length * (a_avg)))
                pt2_avg = (int(x0_avg - line_length * (-b_avg)), int(y0_avg - line_length * (a_avg)))

                # Draw the averaged line on the image
                cv2.line(binary_image, pt1_avg, pt2_avg, (255, 0, 0), 1, cv2.LINE_AA)  # Blue line for average
                cv2.line(display_frame1, pt1_avg, pt2_avg, (255, 0, 0), 1, cv2.LINE_AA)  # Blue line for average
                print("point2 :", pt2_avg , " point1" ,pt1_avg )
                #finding y coodinate of the line for object_x pixel
                # Print or return the midpoint
                #print("Midpoint of the averaged line:", midpoint_avg)
               
        earth_radius = 6371008.77  # Radius of Earth in meters
        camera_fy = 1675        # Example focal length of the camera in pixels
        camera_height = 2.5      # Camera height above ground in meters

        # Convert dilated image to BGR for stacking
        dilated_image_bgr = cv2.cvtColor(dilated_image, cv2.COLOR_GRAY2BGR)

        distance_buoy = calculate_distance(716.113185,520,pt1_avg[0],pt1_avg[1],pt2_avg[0],pt2_avg[1],earth_radius, camera_fy, camera_height)
        print("object distance :", distance_buoy)

        # # Stack the images in a 2x2 matrix layout
        # top_row = np.hstack((frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
        # bottom_row = np.hstack((stable_frame, binary_image))
        # combined = np.vstack((top_row, bottom_row))

        # Write the combined frame to the output video
        out.write(display_frame1)

        # Display the 2x2 matrix of images
        cv2.imshow("video with horizon line", display_frame1)
        #cv2.imshow(stable_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the video capture and writer, and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
detect_horizontal_lines_in_video('MAH01462.mp4', 'output_video_test.mp4')  # Replace with the path to your video and desired output path
