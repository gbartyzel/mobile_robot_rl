import numpy as np

ROBOT_MODEL = {
    'wheel_diameter': 0.06,
    'body_width': 0.156,
    'min_velocity': 0.0,
    'max_velocity': 10.0,
    'robot_objects': {
        'left_motor': 'smartBotLeftMotor',
        'right_motor': 'smartBotRightMotor',
        'robot': 'smartBot'
    },
    'robot_streams': {
        'proximity': 'distanceSignal',
        'accelerometer': 'accelSignal',
        'gyroscope': 'gyroSignal',
        'left_encoder': 'leftEncoder',
        'right_encoder': 'rightEncoder'
    }
}


def square_goals():
    step_size = 0.05
    width = 8.0
    height = 8.0
    center = (-2.0, -2.0)

    angles = np.arange(0.0, 2 * np.pi + step_size, step_size)
    x_es = np.reshape(4 * (np.abs(np.cos(angles)) * np.cos(angles) +
                           np.abs(np.sin(angles)) * np.sin(angles)) - 2.0,
                      (-1, 1))
    y_es = np.reshape(4 * (np.abs(np.cos(angles)) * np.cos(angles) -
                           np.abs(np.sin(angles)) * np.sin(angles)) - 2.0,
                      (-1, 1))

    return np.round(np.append(x_es, y_es, axis=1), 2).tolist()


ENV_SCENES = {
    'no_obstacles': {
        'scene_file': ('no_obstacles.ttt', ) * len(square_goals()),
        'target_position': square_goals(),
        'steps': 1200,
        'dt': 0.05,
        'action_dim': 2,
        'observation_dim': 7,
        'robot_model': ROBOT_MODEL,
    },
    'room': {
        'scene_file': ('room.ttt', ),
        'target_position': ((2.0, 2.0), ),
        'steps': 1200,
        'dt': 0.05,
        'action_dim': 2,
        'observation_dim': 7,
        'robot_model': ROBOT_MODEL,
    },
    'room_four_goals': {
        'scene_file': ('room_four_goals_1.ttt', 'room_four_goals_2.ttt',
                       'room_four_goals_3.ttt', 'room_four_goals_4.ttt'),
        'target_position': ((2.0, 2.0), (2.0, -2.0), (-2.0, -2.0), (-2.0,
                                                                    2.0)),
        'steps':
        1200,
        'dt':
        0.05,
        'action_dim':
        2,
        'observation_dim':
        7,
        'robot_model':
        ROBOT_MODEL,
    },
    'moving_obstacles': {
        'scene_file': ('moving_obstacles.ttt', ),
        'target_position': ((2.0, 2.0), ),
        'steps': 1200,
        'dt': 0.05,
        'action_dim': 2,
        'observation_dim': 7,
        'robot_model': ROBOT_MODEL,
    }
}
