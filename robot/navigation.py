import numpy as np


class Navigation(object):
    """
    Computing navigation parameters of the robots like: (x,y,theta) position,
    path done by robot, error distance and error angule between robot and
    target.
    """
    def __init__(self,
                 start_position,
                 target_position,
                 wheel_diameter,
                 robot_width):

        self.position = np.asarray(start_position)
        self.target_pos = np.asarray(target_position)

        # index 0 - x position
        # index 1 - y position
        # index 2 - theta rotation
        self.nav_data = np.zeros(2)

        self.prev_phi = np.zeros(2)
        self.delta_phi = np.zeros(2)
        self.delta_path = 0
        self.sum_path = 0
        self.previous_av = 0

        self.delta_beta = 0.0
        self.beta = 0.0
        self.wheel_radius = wheel_diameter / 2.0
        self.body_width = robot_width

    @property
    def position(self):
        return self.position

    @property
    def navigation_error(self):
        return np.round(self.nav_data, 3)

    @property
    def path(self):
        return self.sum_path

    @property
    def rotation(self):
        return self.position[2]

    @property
    def delta_theta(self):
        return self.delta_beta

    def compute_delta_phi(self, phi):
        if not phi:
            phi = self.prev_phi
        else:
            phi = np.asarray(phi)
            phi = np.round(phi, 6)

        self.delta_phi = phi - self.prev_phi
        self.prev_phi = phi
        return None

    def compute_position(self, phi, av, position):

        self.compute_delta_phi(phi)
        self.compute_path()
        # self.compute_gyro_rotation(av)

        # self.position += np.array([
        #     self.delta_path * np.cos(self.position[2] + self.delta_beta / 2),
        #     self.delta_path * np.sin(self.position[2] + self.delta_beta / 2),
        #     self.delta_beta
        #     ])
        self.position = position[0:2] + [position[5]]
        self.position = np.round(self.position, 3)
        self.position[2] = self._angle_correction(self.position[2])

        temp = np.sqrt(np.sum((self.target_pos - self.position[0:2])**2))

        if not np.isnan(temp):
            self.nav_data[0] = temp

        theta = np.arctan2(self.target_pos[1] - self.position[1],
                           self.target_pos[0] - self.position[0])

        theta = self._angle_correction(theta)

        self.nav_data[1] = self._angle_correction(theta - self.position[2])

    def compute_path(self):

        wheels_paths = self.delta_phi * self.wheel_radius

        self.delta_path = np.round(np.sum(wheels_paths) / 2, 5)
        self.sum_path += self.delta_path

    def compute_rotation(self):
        wheels_paths = self.delta_phi * self.wheel_radius

        self.delta_beta = (wheels_paths[1] - wheels_paths[0]) / self.body_width
        self.delta_beta = np.round(self.delta_beta, 6)

    def compute_gyro_rotation(self, current_av):
        self.delta_beta = current_av * 0.1
        self.delta_beta = np.round(self.delta_beta, 6)
        self.previous_av = current_av

    def set_target_position(self, target_position):
        self.target_pos = np.asarray(target_position)

    @staticmethod
    def _angle_correction(angle):
        if angle >= 0:
            angle = np.mod((angle + np.pi), (2 * np.pi)) - np.pi

        if angle < 0:
            angle = np.mod((angle - np.pi), (2 * np.pi)) + np.pi

        return np.round(angle, 6)
