# KeyFrameMsg.msg

# Height map (H, W) as an Image
sensor_msgs/Image height_map
# Contact mask (H, W) as an Image
sensor_msgs/Image contact_mask
# Normal map (H, W, 3) as an Image
sensor_msgs/Image normal_map
# Laplacian map (H, W) as an Image
sensor_msgs/Image laplacian_map

# Keyframe's ID number
int32 kid
# Keyframe's Frame ID number
int32 fid
# New trial flag
bool new_trial_flag
# Transformation matrix (4, 4) as a flattened array
float32[] ref_t_curr
