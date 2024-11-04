import cv2
import math
import numpy as np
kernelErode = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 28))

kerneDilate = np.ones((30, 30), dtype=np.uint8)
color1 = (4, 7, 139)[::-1]
color2 = (143, 154, 189)[::-1]

def calculate_distance(px_obj,py_obj,horizon_ox1,horizon_oy1,horizon_ox2,horizon_oy2,earth_radius, camera_fy, camera_height):
    line_y = (((horizon_oy2-horizon_oy1)/(horizon_ox2-horizon_ox1))*px_obj + horizon_oy1)
    #line_y = (horizon_oy2+horizon_oy1)/2
    pixel_distance= math.sqrt((px_obj - px_obj) ** 2 + (line_y - py_obj) ** 2)
    theta_dy = math.atan(pixel_distance / camera_fy)
    principle_line = math.sqrt(2*camera_height*earth_radius + camera_height*camera_height)
    # horizon_dist = math.sqrt(principle_line*principle_line-camera_height*camera_height)
    theta_c = math.asin(camera_height/principle_line)
    distance_to_object = camera_height/math.tan(theta_c+theta_dy)

    return distance_to_object


# def calculate_distance(px_obj, py_obj, horizon_ox1, horizon_oy1, horizon_ox2, horizon_oy2, earth_radius, camera_fy, camera_height):
    # Correct the calculation of line_y to ensure it reflects the actual vertical position of the horizon
    if horizon_ox2 != horizon_ox1:  # Prevent division by zero
        line_y = (((horizon_oy2 - horizon_oy1) / (horizon_ox2 - horizon_ox1)) * (px_obj - horizon_ox1) + horizon_oy1)
    else:
        line_y = (horizon_oy2 + horizon_oy1) / 2  # Fallback to the average if line is vertical

    # Correct pixel distance calculation (vertical distance only)
    pixel_distance = abs(line_y - py_obj)

    # Calculate the angle theta_dy
    theta_dy = math.atan(pixel_distance / camera_fy)

    # Compute the principle line and theta_c
    principle_line = math.sqrt(2 * camera_height * earth_radius + camera_height ** 2)
    theta_c = math.asin(camera_height / principle_line)

    # Calculate the distance to the object
    distance_to_object = camera_height / math.tan(theta_c + theta_dy)

    return distance_to_object

def detect_horizontal_lines_in_video(frame, x_coor, y_coor):
    distance_buoy=0
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
    eroded_image = cv2.erode(dilated_image,kernelErode,1)

    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLines(eroded_image, 1, np.pi/180, 850, None, 0, 0) #best ever for stabilized
    horizontal_lines = np.zeros_like(frame)
    total_rho = 0
    total_theta = 0
    count = 0
    height, width = frame.shape[:2]
    line_length = 2000

    # Initialize variables for the previous average line
    prev_avg_rho = None
    prev_avg_theta = None
    earth_radius = 6378137  # Radius of Earth in meters
    camera_fy = 1675        # Example focal length of the camera in pixels
    camera_height = 2.5      # Camera height above ground in meters

    if lines is not None:
        total_rho = 0
        total_theta = 0
        count = 0
        
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            # Line filter based on angle and distance
            if abs(90 - np.degrees(theta)) < 1 and 200 < rho < 600:
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + line_length * (-b)), int(y0 + line_length * (a)))
                pt2 = (int(x0 - line_length * (-b)), int(y0 - line_length * (a)))

                cv2.line(dilated_image, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(frame, pt1, pt2, color2, 1, cv2.LINE_AA)
                
                # Accumulate rho and theta
                total_rho += rho
                total_theta += theta
                count += 1

        # Calculate the average line if any lines were found
        if count > 0:
            avg_rho = total_rho / count
            avg_theta = total_theta / count

            # alpha = 0.4  # Smoothing factor
            # if prev_avg_rho is not None and prev_avg_theta is not None:
            #     avg_rho = alpha * avg_rho + (1 - alpha) * prev_avg_rho
            #     avg_theta = alpha * avg_theta + (1 - alpha) * prev_avg_theta

            # Update the previous average line variables
            prev_avg_rho = avg_rho
            prev_avg_theta = avg_theta

            # Calculate endpoints of the averaged line
            a_avg = math.cos(avg_theta)
            b_avg = math.sin(avg_theta)
            x0_avg = a_avg * avg_rho
            y0_avg = b_avg * avg_rho
            pt1_avg = (int(x0_avg + line_length * (-b_avg)), int(y0_avg + line_length * (a_avg)))
            pt2_avg = (int(x0_avg - line_length * (-b_avg)), int(y0_avg - line_length * (a_avg)))

            # Draw the averaged line on the image
            cv2.line(dilated_image, pt1_avg, pt2_avg, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.line(frame, pt1_avg, pt2_avg, color1, 1, cv2.LINE_AA)

            # Calculate distance to buoy
            distance_buoy = calculate_distance(x_coor, y_coor, pt1_avg[0], pt1_avg[1], pt2_avg[0], pt2_avg[1], earth_radius, camera_fy, camera_height)

    # If no new lines were found, draw the previous averaged line
    elif prev_avg_rho is not None and prev_avg_theta is not None:
        # Calculate endpoints of the last averaged line
        a_avg = math.cos(prev_avg_theta)
        b_avg = math.sin(prev_avg_theta)
        x0_avg = a_avg * prev_avg_rho
        y0_avg = b_avg * prev_avg_rho
        pt1_avg = (int(x0_avg + line_length * (-b_avg)), int(y0_avg + line_length * (a_avg)))
        pt2_avg = (int(x0_avg - line_length * (-b_avg)), int(y0_avg - line_length * (a_avg)))

        # Draw the previous averaged line on the image
        cv2.line(dilated_image, pt1_avg, pt2_avg, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(frame, pt1_avg, pt2_avg, color1, 1, cv2.LINE_AA)

        # Calculate distance to buoy using previous line
        distance_buoy = calculate_distance(x_coor, y_coor, pt1_avg[0], pt1_avg[1], pt2_avg[0], pt2_avg[1], earth_radius, camera_fy, camera_height)


    eroded_image_bgr =cv2.cvtColor(eroded_image, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("video with horizon line", frame)
    # cv2.imshow("eroded", eroded_image_bgr)
    
    return distance_buoy 
