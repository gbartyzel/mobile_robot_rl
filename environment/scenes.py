import numpy as np

ROBOT_MODEL = {
    "wheel_diameter": 0.06,
    "body_width": 0.156,
    "min_velocity": 0.0,
    "max_velocity": 10.0,
    "robot_objects": {
        "left_motor": "smartBotLeftMotor",
        "right_motor": "smartBotRightMotor",
        "robot": "smartBot"
    },
    "robot_streams": {
        "proximity": "distanceSignal",
        "accelerometer": "accelSignal",
        "gyroscope": "gyroSignal",
        "left_encoder": "leftEncoder",
        "right_encoder": "rightEncoder"
    }
}

ENV_SCENES = {
    "no_obstacles": {
        "scene_file": "no_obstacles.ttt",
        "target_position": np.round((np.random.rand(2) - 1) * 2),
        "steps": 2400,
        "dt": 0.05,
        "action_dim": 2,
        "observation_dim": 7,
        "robot_model": ROBOT_MODEL,
    },
    "room": {
        "scene_file": "room.ttt",
        "target_position": (2.0, 2.0),
        "steps": 2400,
        "dt": 0.05,
        "action_dim": 2,
        "observation_dim": 7,
        "robot_model": ROBOT_MODEL,
    },
    "room_random": {
        "scene_file": ("room_random_1.ttt", "room_random_2.ttt",
                       "room_random_3.ttt", "room_random_4.ttt"),
        "target_position": ((2.0, 2.0), (2.0, -2.0), (-2.0, -2.0), (-2.0,
                                                                    2.0)),
        "steps":
        2400,
        "dt":
        0.05,
        "action_dim":
        2,
        "observation_dim":
        7,
        "robot_model":
        ROBOT_MODEL,
    },
    "moving_obstacles": {
        "scene_file": "moving_obstacles.ttt",
        "target_position": (2.0, 2.0),
        "steps": 2400,
        "dt": 0.05,
        "action_dim": 2,
        "observation_dim": 7,
        "robot_model": ROBOT_MODEL,
    }
}
