import numpy as np

from base import Base


class Gyrodometry(Base):
    def __init__(self, start_pose, target_position, wheel_diameter,
                 robot_width, dt):
        super(Gyrodometry, self).__init__(target_position, wheel_diameter,
                                          robot_width, dt)

        self._pose = start_pose
        self._sum_path = 0.0

        self._previous_phi = 0.0
        self._previous_angular_velocity = 0.0

    @property
    def sum_path(self):
        return self._sum_path

    def compute_position(self, **kwargs):
        delta_path = self.compute_delta_motion(kwargs['phi'])
        delta_beta = self.compute_rotation(kwargs['anuglar_velocity'])

        self._pose += np.array([
            delta_path * np.cos(self._pose[2] + delta_beta / 2),
            delta_path * np.sin(self._pose[2] + delta_beta / 2), delta_beta
        ])

        self._pose = np.round(self._pose, 3)
        self._pose[2] = self._angle_correction(self._pose[2])

        self._navigation_error[0] = np.sqrt(
            np.sum((self._target_position - self._pose[0:2])**2))

        theta = np.arctan2(self._target_position[1] - self._pose[1],
                           self._target_position[0] - self._pose[0])

        theta = self._angle_correction(theta)

        self._navigation_error[1] = self._angle_correction(
            theta - self._pose[2])

    def compute_delta_phi(self, phi):
        if not phi:
            phi = self._previous_phi
        else:
            phi = np.asarray(phi)
            phi = np.round(phi, 3)

        delta_phi = phi - self._previous_phi
        self._previous_phi = phi

        return delta_phi

    def compute_delta_motion(self, phi):
        delta_phi = self.compute_delta_phi(phi)
        wheels_paths = delta_phi * self._wheel_radius

        delta_path = np.round(np.sum(wheels_paths) / 2, 3)
        self._sum_path += delta_path

        return delta_path

    def compute_rotation(self, current_angular_velocity):
        delta_beta = np.round(current_angular_velocity * self._dt, 3)
        self._previous_angular_velocity = current_angular_velocity

        return delta_beta
