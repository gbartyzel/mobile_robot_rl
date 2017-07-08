import numpy as np
from vrep import vrep


class Robot(object):
    """
    Robot class for V-REP smarbot model
    """
    def __init__(self,
                 client,
                 diameter,
                 width,
                 dt,
                 velocity_bound,
                 streams_names,
                 objects_names):
        """
        :param client: v-rep connection client ID
        :param diameter: robot wheel diameter
        :param width: robot body width (distance between wheels)
        :param dt: time of one simulation step in sec
        :param velocity_bound: absolute value of velocity bound for motors
        :param streams_names: list of v-rep model's streams:
                              0 - proximity sensor signal,
                              1 - accelerometer signal,
                              2 - gyroscope signal,
                              3 - left encoder signal,
                              4 - right encoder signal
        :param objects_names: list of v-rep model's objects:
                              0 - lef motor handler,
                              1 - right motor handler,
                              2 - body handler
        """

        self.objects_names = objects_names
        self.streams_names = streams_names
        
        self.objects_handlers = []

        self.total_phi = [0.0, 0.0]
        self.dphi = [0.0, 0.0]

        self._client_id = client
        self._wheel_diameter = diameter
        self._body_width = width
        self._dt = dt
        self._vel_bound = velocity_bound

        self.gyroscope_data = 0.0
        self.accelerometer_data = np.array([0.0, 0.0, 0.0])

        self.initialize()

        self.set_motor_velocities([0.0, 0.0])

    def initialize(self):
        """
        Initialize connection with objects and streams of v-rep model.
        :return: nothing to return
        """
        for object_n in self.objects_names:
            res, temp = vrep.simxGetObjectHandle(
                self._client_id, object_n, vrep.simx_opmode_oneshot_wait)
            self.objects_handlers.append(temp)

        for i in range(5):
            if i in range(2):
                res, temp = vrep.simxReadStringStream(
                    self._client_id, self.streams_names[i],
                    vrep.simx_opmode_streaming)
            else:
                res, temp = vrep.simxGetFloatSignal(
                    self._client_id, self.streams_names[i],
                    vrep.simx_opmode_streaming)

    @property
    def wheel_diameter(self):
        """
        :return: robot wheel diameter 
        """
        return self._wheel_diameter

    @property
    def body_width(self):
        """
        :return: robot body width
        """
        return self._body_width


    def read_encoders(self):
        """
        Read temporary rotation of the wheels.
        :return: total rotation angle of the wheels
        """
        for i in range(2):
            res, temp = vrep.simxGetFloatSignal(
                self._client_id, self.streams_names[i + 3],
                vrep.simx_opmode_buffer)

            if res == vrep.simx_return_ok:
                self.dphi[i] = temp - self.total_phi[i]
                self.total_phi[i] = temp

        return self.total_phi

    def read_velocities(self):
        """
        Compute wheels velocities.
        :return: wheels velocities
        """
        self.velocities = (np.asarray(self.dphi) / self._dt).tolist()
        vel = (np.sum(self.velocities) * self._wheel_diameter / 2) / 2
        return np.round(self.velocities + [vel], 2).tolist()

    def read_proximity_sensors(self):
        """
        :return: distances measured by proximity sensors
        """
        res, packed_vec = vrep.simxReadStringStream(
            self._client_id, self.streams_names[0], vrep.simx_opmode_buffer)

        if res == vrep.simx_return_ok:
            unpacked_vec = vrep.simxUnpackFloats(packed_vec)[0:5]

            self.proximity_data = self._noise_model(unpacked_vec, 0.005)
            self.proximity_data = np.round(self.proximity_data, 3)

            self.proximity_data[self.proximity_data > 2.0] = 2.0
            self.proximity_data[self.proximity_data < 0.02] = 0.02

        return self.proximity_data

    def read_accelerometer(self):
        """
        :return: absolute robot acceleration measured by accelerometer.
        """
        res, packed_vec = vrep.simxReadStringStream(
            self._client_id, self.streams_names[1], vrep.simx_opmode_buffer)

        if res == vrep.simx_return_ok:
            unpacked_vec = vrep.simxUnpackFloats(packed_vec)[0:3]
            self.accelerometer_data = self._noise_model(unpacked_vec, 0.005)

        return self.accelerometer_data

    def read_gyroscope(self):
        """
        :return: robot rotation around Z axies measured by gyroscope
        """
        res, self.gyroscope_data = vrep.simxGetFloatSignal(
            self._client_id, self.streams_names[2], vrep.simx_opmode_buffer)

        # if res == vrep.simx_return_ok:
        #    self.gyroscope_data = self._noise_model(received_data, 0.005)
        return self.gyroscope_data

    def set_motor_velocities(self, velocities):
        """
        Method that set target velocities.
        :param velocities: target velocities
        """
        if isinstance(velocities, list):
            velocities = np.asarray(velocities)
        velocities = np.round(velocities, 2)
        velocities[velocities > self._vel_bound] = self._vel_bound
        velocities[velocities < -self._vel_bound] = -self._vel_bound
        velocities = np.round(velocities, 2)
        for i, vel in enumerate(velocities):
            vrep.simxSetJointTargetVelocity(
                self._client_id, self.objects_handlers[i], vel,
                vrep.simx_opmode_oneshot_wait)

    def position(self):
        """
        The output od this method is absolute robot position in simulated
        environment.
        :return: absolute robot position and orientation
        """
        res, pos = vrep.simxGetObjectPosition(
            self._client_id, self.objects_handlers[2], -1,
            vrep.simx_opmode_oneshot)

        res, rot = vrep.simxGetObjectOrientation(
            self._client_id, self.objects_handlers[2], -1,
            vrep.simx_opmode_oneshot)

        return np.round(pos + [rot[2]], 5)

    @staticmethod
    def _noise_model(data, diff):
        """
        Method that transfer noiseless signal from simulator to signal with
        noise.
        :param data: noiseless signal 
        :param diff: standard deviation
        :return: noised signal
        """
        noise_data = np.asarray(data) + np.random.uniform(-diff, diff)
        return noise_data
