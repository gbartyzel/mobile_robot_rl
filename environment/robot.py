import numpy as np
from environment.vrep import vrep


class Robot(object):
    def __init__(self,
                 client,
                 streams_names,
                 objects_names,
                 diameter=0.06,
                 width=0.154,
                 dt=0.05,
                 min_velocity=0.0,
                 max_velocity=10.0):
        self._streams_names = streams_names

        self._objects_hanlders = objects_names.copy()

        self._total_phi = [0.0, 0.0]
        self._dphi = [0.0, 0.0]

        self._client_id = client
        self._wheel_diameter = diameter
        self._body_width = width
        self._dt = dt
        self._vel_bound = [min_velocity, max_velocity]

        self.initialize(objects_names)

        self.set_motor_velocities([0.0, 0.0])

    def initialize(self, objects_names):
        for key, name in objects_names.items():
            res, temp = vrep.simxGetObjectHandle(self._client_id, name,
                                                 vrep.simx_opmode_oneshot_wait)
            self._objects_hanlders[key] = temp

        for key, stream in self._streams_names.items():
            if key == 'proximity' or key == 'accelerometer':
                vrep.simxReadStringStream(self._client_id, stream,
                                          vrep.simx_opmode_streaming)
            else:
                vrep.simxGetFloatSignal(self._client_id, stream,
                                        vrep.simx_opmode_streaming)

    def set_motor_velocities(self, velocities):
        if isinstance(velocities, list):
            velocities = np.asarray(velocities)

        velocities[velocities < self._vel_bound[0]] = self._vel_bound[0]
        velocities[velocities > self._vel_bound[1]] = self._vel_bound[1]

        vrep.simxSetJointTargetVelocity(
            self._client_id, self._objects_hanlders['left_motor'],
            velocities[0], vrep.simx_opmode_oneshot_wait)

        vrep.simxSetJointTargetVelocity(
            self._client_id, self._objects_hanlders['right_motor'],
            velocities[1], vrep.simx_opmode_oneshot_wait)

    @property
    def wheel_diameter(self):
        return self._wheel_diameter

    @property
    def body_width(self):
        return self._body_width

    @property
    def vel_bound(self):
        return self._vel_bound

    def get_encoders_values(self):
        for i, key in enumerate(['left_encoder', 'right_encoder']):
            _, ticks = vrep.simxGetFloatSignal(self._client_id,
                                               self._streams_names[key],
                                               vrep.simx_opmode_buffer)

            self._dphi[i] = ticks - self._total_phi[i]
            self._total_phi[i] = ticks

        return self._total_phi

    def get_velocities(self):
        velocities = (np.asarray(self._dphi) / self._dt).tolist()
        vel = (np.sum(velocities) * self._wheel_diameter / 2) / 2
        return np.round(velocities + [vel], 2).tolist()

    def get_proximity_values(self):
        _, packed_vec = vrep.simxReadStringStream(
            self._client_id, self._streams_names['proximity'],
            vrep.simx_opmode_buffer)

        data = vrep.simxUnpackFloats(packed_vec)[0:5]

        data = np.round(self._noise_model(data, 0.005), 3)

        data[data > 2.0] = 2.0
        data[data < 0.02] = 0.02

        return data

    def get_accelerometer_values(self):
        _, packed_vec = vrep.simxReadStringStream(
            self._client_id, self._streams_names['acceletometer'],
            vrep.simx_opmode_buffer)

        data = vrep.simxUnpackFloats(packed_vec)

        return self._noise_model(data, 0.005)

    def get_gyroscope_values(self):
        _, data = vrep.simxGetFloatSignal(self._client_id,
                                          self._streams_names['gyroscope'],
                                          vrep.simx_opmode_buffer)

        return self._noise_model(data, 0.005)

    def get_position(self):
        _, pos = vrep.simxGetObjectPosition(self._client_id,
                                            self._objects_hanlders['robot'],
                                            -1, vrep.simx_opmode_oneshot)

        _, rot = vrep.simxGetObjectOrientation(self._client_id,
                                               self._objects_hanlders['robot'],
                                               -1, vrep.simx_opmode_oneshot)

        return np.round(pos[0:2] + [rot[2]], 5)

    @staticmethod
    def _noise_model(data, diff):
        noise_data = np.asarray(data) + np.random.uniform(-diff, diff)
        return noise_data
