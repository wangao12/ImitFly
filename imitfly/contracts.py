"""Ordered inputs of the v51 checkpoint; FlyBody uses cm, g, seconds."""
SENSOR_DIMS = {
    "accelerometer": 3, "gyro": 3, "joints_pos": 25, "joints_vel": 25,
    "velocimeter": 3, "world_zaxis": 3, "task_input": 2,
}
ACTION_DIM = 12
IMAGE_SIZE = 32
SEQUENCE_LENGTH = 5

WALKING_SENSOR_DIMS = {'accelerometer': 3, 'gyro': 3, 'velocimeter': 3, 'world_zaxis': 3, 'joints_pos': 85, 'joints_vel': 85, 'actuator_activation': 59, 'appendages_pos': 21, 'force': 18, 'touch': 6, 'ref_displacement': 195, 'ref_root_quat': 260}
