import yaml
from ESEKF_IMU.esekf import *


def load_imu_parameters(config='./data/params.yaml'):
    with open(config, 'r') as f:
        yml = yaml.safe_load(f)
        params = ImuParameters()
        params.frequency = yml['IMU.frequency']
        params.sigma_a_n = yml['IMU.acc_noise_sigma']  # m/sqrt(s^3)
        params.sigma_w_n = yml['IMU.gyro_noise_sigma']  # rad/sqrt(s)
        params.sigma_a_b = yml['IMU.acc_bias_sigma']  # m/sqrt(s^5)
        params.sigma_w_b = yml['IMU.gyro_bias_sigma']  # rad/sqrt(s^3)
        params.Gravity = yml['Gravity']  # m/s^2

    return params


def init_estimator(gps0, imu0, imu_parameters, sigma_p=0.02, sigma_q=0.015):
    init_nominal_state = np.zeros((19,))
    init_nominal_state[:10] = gps0[1:11]  # init p, q, v
    init_nominal_state[10:13] = 0  # init ba
    init_nominal_state[13:16] = 0  # init bg
    init_nominal_state[16:19] = np.array(imu_parameters.Gravity)  # init g
    start_time = imu0[0]
    estimator = ESEKF(init_nominal_state, imu_parameters, start_time)

    sigma_measurement_p = sigma_p  # in meters
    sigma_measurement_q = sigma_q  # in rad
    sigma_measurement = np.eye(6)
    sigma_measurement[0:3, 0:3] *= sigma_measurement_p ** 2
    sigma_measurement[3:6, 3:6] *= sigma_measurement_q ** 2

    return estimator, sigma_measurement


def ekf_predict(estimator, imu_data):
    timestamp = imu_data[0]
    estimator.predict(imu_data)

    print('EKF predict [%f]:' % timestamp, estimator.nominal_state)
    frame_pose = np.zeros(8, )
    frame_pose[0] = timestamp
    frame_pose[1:] = estimator.nominal_state[:7]

    return frame_pose


def ekf_update(estimator, gps_data, sigma_measurement):
    timestamp = gps_data[0]
    estimator.update(gps_data[1:8], sigma_measurement)  # update filter by measurement.

    print('EKF update [%f]:' % timestamp, estimator.nominal_state)
    frame_pose = np.zeros(8, )
    frame_pose[0] = timestamp
    frame_pose[1:] = estimator.nominal_state[:7]  # timestamp x y z qx qy qz qw (TUM format)

    return frame_pose
