# FrameMsg.msg

# Height map (H, W) as an Image
sensor_msgs/Image height_map
# Contact mask (H, W) as an Image
sensor_msgs/Image contact_mask
# Normal map (H, W, 3) as an Image
sensor_msgs/Image normal_map
# Laplacian map (H, W) as an Image
sensor_msgs/Image laplacian_map

# Frame's ID number
int32 fid
# Reference keyframe's ID number
int32 ref_kid
# Transformation matrix (4, 4) as a flattened array
float32[] ref_t_curr
