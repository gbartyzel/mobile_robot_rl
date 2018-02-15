import numpy as np

ENV_SCENES = {
    "no_obstacles": {
        "scene_file": "no_obstacles.ttt",
        "target_position": np.round((np.random.rand(2) - 1) * 2),
        "steps": 2400,
        "dt": 0.05,
    },
    "room": {
        "scene_file": "room.ttt",
        "target_position": (2.0, 2.0),
        "steps": 2400,
        "dt": 0.05,
    },
    "room_random": {
        "scene_file": ("room_random_1.ttt", "room_random_2.ttt",
                       "room_random_3.ttt", "room_random_4.ttt"),
        "target_position": ((2.0, 2.0), (2.0, -2.0), (-2.0, -2.0), (-2.0,
                                                                    2.0)),
        "steps":
        2400,
        "dt":
        0.05
    },
    "moving_obstacles": {
        "scene_file": "moving_obstacles.ttt",
        "target_position": (2.0, 2.0),
        "steps": 2400,
        "dt": 0.05
    }
}
