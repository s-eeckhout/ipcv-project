#depth detection file
import math

#below function accepts pixels of the object and the pixels of horizon which are camera principal point 
#camera_heigh is 2.5m
#earth radius is 6371008.77m
#camera_fy is focal lengh for the y axis of camera

def calculate_distance(px_obj,py_obj,camera_ox,camera_oy,earth_radius, camera_fy, camera_height):
    pixel_distance= math.sqrt((camera_ox - px_obj) ** 2 + (camera_oy - py_obj) ** 2)
    theta_dy = math.atan(pixel_distance / camera_fy)
    principle_line = math.sqrt(2*camera_height*earth_radius + camera_height*camera_height)
    horizon_dist = math.sqrt(principle_line*principle_line-camera_height*camera_height)
    theta_c = math.asin(camera_height/horizon_dist)
    distance_to_object = camera_height/math.tan(theta_c+theta_dy)

    return distance_to_object

earth_radius = 6371008.77  # Radius of Earth in meters
camera_fy = 1675        # Example focal length of the camera in pixels
camera_height = 2.5      # Camera height above ground in meters

oy=554.7891784
ox=716.113185

obj_x = 716.113185
obj_y = 520

distance = calculate_distance(obj_x,obj_y,ox,oy,earth_radius, camera_fy, camera_height)
print(f"Estimated distance to object: {distance:.2f} meters")


# Intrinsic matrix 		
# 1252.349447	0	716.113185
# 0	1666.6412	554.7891784
# 0	0	1