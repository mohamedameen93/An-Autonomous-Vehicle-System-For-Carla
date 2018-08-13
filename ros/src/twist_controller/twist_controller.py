import rospy

from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, wheel_radius, steer_ratio, max_lat_accel, max_steer_angle,
                 decel_limit, vehicle_mass):
        self.yaw_controller = YawController(
            wheel_base,
            steer_ratio,
            0.1,  # min_speed
            max_lat_accel,
            max_steer_angle,
        )

        # PID coefficients
        kp = 0.3
        ki = 0.1
        kd = 0.0

        # For convenience (no real limits on throttle):
        mn_th = 0.0  # Minimal throttle
        mx_th = 0.2  # Maximal throttle
        self.throttle_controller = PID(kp, ki, kd, mn_th, mx_th)

        tau = 0.5  # 1 / (2pi*tau) = cutoff frequency
        ts = 0.02  # Sample time
        self.vel_lpf = LowPassFilter(tau, ts)

        self.wheel_radius = wheel_radius
        self.decel_limit = decel_limit
        self.vehicle_mass = vehicle_mass

        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0

        filtered_vel = self.vel_lpf.filt(current_vel)

        # rospy.logwarn(
        #     '\nAngular vel: {}\n'
        #     'Target velocity: {}\n'
        #     'Current velocity: {}\n'
        #     'Filtered velocity: {}\n'
        #     .format(angular_vel, linear_vel, current_vel, filtered_vel)
        # )

        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, filtered_vel)

        vel_error = linear_vel - filtered_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)

        brake = 0

        if linear_vel == 0 and filtered_vel < 0.1:
            throttle = 0
            brake = 400  # [N*m] -- to hold the car in place if we are stopped
                         # at a light. Acceleration ~ 1 m/s/s
        elif throttle < 0.1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius  # Torque [N*m]

        return throttle, brake, steering
