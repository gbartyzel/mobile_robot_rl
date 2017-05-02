import numpy as np

try:
    import vrep
except:
    print('Error!')


class Robot(object):
    """"
    Robot class for V-REP smarbot model
    """
    def __init__(self,  client, diameter, width, streams_names, objects_names):
        """
        :param client: v-rep connection client ID 
        :param diameter: robot wheel diameter
        :param width: robot body width (distance between wheels)
        :param streams_names: names of v-rep model streams
        :param objects_names: name of v-rep model objects
        """
        self.client_id = client
        self.wheel_diameter = diameter
        self.body_width = width
        self.dt = 0.1
        self.objects_names = objects_names
        self.streams_names = streams_names
        # 0 - proximity sensor signal, 1 - accelerometer signal
        # 2 - gyroscope signal
        self.objects_handlers = []
        # 0 - left motor handler, 1 - right motor handler, 2 = body handler

        self.total_phi = [0.0, 0.0]
        self.velocities = [0.0, 0.0]
        self.dphi = [0.0, 0.0]
        self.proximity_data = np.zeros(5).tolist()
        self.gyroscope_data = 0.0
        self.accelerometer_data = [0.0, 0.0, 0.0]

        self.initialize()

        self.set_motor_velocities(self.velocities)

    @staticmethod
    def _noise_model(clean_input, std_dev):
        """
        Method that transfer noiseless signal from simulator to signal with
        noise.
        :param clean_input: noiseless signal 
        :param std_dev: standard deviation
        :return: noised signal
        """
        noise_data = np.random.normal(clean_input, std_dev)
        return noise_data.tolist()

    def initialize(self):
        """
        Initialize connection with objects and streams of v-rep model.
        :return: nothing to return
        """
        for object_n in self.objects_names:
            res = 0xFFFFFF

            while res != vrep.simx_return_ok:
                res, temp = vrep.simxGetObjectHandle(
                    self.client_id, object_n, vrep.simx_opmode_oneshot_wait)
                self.objects_handlers.append(temp)

        for i in range(5):
            res = 0xFFFFFF
            while res != vrep.simx_return_ok:
                if i in range(2):
                    res, temp = vrep.simxReadStringStream(
                        self.client_id, self.streams_names[i],
                        vrep.simx_opmode_streaming)
                else:
                    res, temp = vrep.simxGetFloatSignal(
                        self.client_id, self.streams_names[i],
                        vrep.simx_opmode_streaming)
        return None

    def reset(self):
        self.total_phi = [0.0, 0.0]
        self.dphi = [0.0, 0.0]
        self.velocities = [0.0, 0.0]
        self.proximity_data = [0.0, 0.0, 0.0, 0.0]
        self.gyroscope_data = 0.0
        self.accelerometer_data = [0.0, 0.0, 0.0]
        self.objects_handlers = []

        self.initialize()

    def read_encoders(self):
        """
        Read temporary rotation of the wheels.
        :return: total rotation angle of the wheels. 
        """
        for i in range(2):
            res, temp = vrep.simxGetFloatSignal(
                self.client_id, self.streams_names[i + 3],
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
        self.velocities = (np.asarray(self.dphi) / self.dt).tolist()
        vel = (np.sum(self.velocities) * self.wheel_diameter / 2) / 2
        return np.round(self.velocities + [vel], 2).tolist()

    def read_proximity_sensors(self):
        """
        :return: distances measured by proximity sensors 
        """
        res, packed_vec = vrep.simxReadStringStream(self.client_id,
                                                    self.streams_names[0],
                                                    vrep.simx_opmode_buffer)

        if res == vrep.simx_return_ok:
            unpacked_vec = vrep.simxUnpackFloats(packed_vec)
            unpacked_vec = unpacked_vec[0:5]

            self.proximity_data = self._noise_model(unpacked_vec, 0.0001)
            self.proximity_data = np.round(self.proximity_data, 3)

            self.proximity_data[self.proximity_data > 2.0] = 2.0
            self.proximity_data[self.proximity_data < 0.02] = 0.02

            return self.proximity_data.tolist()
        else:
            return self.proximity_data.tolist()

    def read_accelerometer(self):
        """
        :return: absolute robot acceleration measured by accelerometer.
        """
        res, packed_vec = vrep.simxReadStringStream(self.client_id,
                                                    self.streams_names[1],
                                                    vrep.simx_opmode_buffer)

        if res == vrep.simx_return_ok:
            unpacked_vec = vrep.simxUnpackFloats(packed_vec)
            unpacked_vec = unpacked_vec[0:3]
            self.accelerometer_data = np.round(unpacked_vec, 3)

            return self.accelerometer_data.tolist()
        else:
            return [0.0, 0.0, 0.0]

    def read_gyroscope(self):
        """
        :return: robot rotation around Z axies measured by gyroscope
        """
        res, temp = vrep.simxGetFloatSignal(self.client_id,
                                            self.streams_names[2],
                                            vrep.simx_opmode_buffer)

        if res == vrep.simx_return_ok:
            self.gyroscope_data = np.round(temp, 4)
            return self.gyroscope_data
        else:
            return self.gyroscope_data

    def set_motor_velocities(self, velocities):
        """
        Method that set target velocities.
        :param velocities: target velocities 
        """
        velocities = np.asarray(velocities)
        velocities = np.round(velocities, 1)
        velocities[velocities > 15.0] = 15.0
        velocities[velocities < -15.0] = -15.0

        for i, velocity in enumerate(velocities):
            vrep.simxSetJointTargetVelocity(self.client_id,
                                            self.objects_handlers[i],
                                            velocity,
                                            vrep.simx_opmode_oneshot_wait)

        return None

    def set_position(self, position, orientation):
        """
        :param position: target robot position 
        :param orientation: target robot rotation
        """
        vrep.simxSetObjectPosition(self.client_id,
                                   self.objects_handlers[2],
                                   -1,
                                   position,
                                   vrep.simx_opmode_oneshot)

        vrep.simxSetObjectOrientation(self.client_id,
                                      self.objects_handlers[2],
                                      -1, orientation,
                                      vrep.simx_opmode_oneshot)

    def get_wheel_diameter(self):
        """
        :return: robot wheel diameter 
        """
        return self.wheel_diameter

    def get_width(self):
        """
        :return: robot body width 
        """
        return self.body_width

    def get_position(self):
        """
        The output od this method is absolute robot position in simulated
        environment.
        :return: absolute robot position and orientation
        """
        res = 0xFFFFFF
        while res != vrep.simx_return_ok:
            res, pos = vrep.simxGetObjectPosition(self.client_id,
                                                  self.objects_handlers[2],
                                                  -1,
                                                  vrep.simx_opmode_oneshot)

        res = 0xFFFFFF
        while res != vrep.simx_return_ok:
            res, rot = vrep.simxGetObjectOrientation(self.client_id,
                                                     self.objects_handlers[2],
                                                     -1,
                                                     vrep.simx_opmode_oneshot)

        pos = np.round(np.asarray(pos), 3).tolist()
        rot = np.round(np.asarray(rot), 3).tolist()
        return pos + rot
