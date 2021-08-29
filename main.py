import numpy as np
from matplotlib import pyplot as plt
from IPython import display
import gpiozero
import time

# Outputs
pwm_l = gpiozero.PWMOutputDevice(pin=12, active_high=True, initial_value=0, frequency=50000)  # motor 1
pwm_r = gpiozero.PWMOutputDevice(pin=14, active_high=True, initial_value=0, frequency=50000)  # motor 2
direction_l = gpiozero.OutputDevice(pin=12)
direction_r = gpiozero.OutputDevice(pin=14)

# Inputs
encoder_l = gpiozero.RotaryEncoder(a=5, b=6, max_steps=100000)
encoder_r = gpiozero.RotaryEncoder(a=7, b=8, max_steps=100000)
sensor_f = gpiozero.DistanceSensor(echo=1, trigger=7)
sensor_l = gpiozero.DistanceSensor(echo=3, trigger=6)
sensor_r = gpiozero.DistanceSensor(echo=4, trigger=5)


# Differential drive robot model
class DiffDriveRobot:

    def __init__(self, inertia=5, dt=0.1, drag=0.2, wheel_radius=0.028, wheel_sep=0.101):
        self.x = 0.0  # y-position
        self.y = 0.0  # y-position
        self.th = 0.0  # orientation in radians
        self.th_deg = 0.0  # orientation in degrees

        self.wl = 0.0  # rotational velocity left wheel
        self.wr = 0.0  # rotational velocity right wheel

        self.inert = inertia
        self.drag = drag
        self.dt = dt

        self.steps_l = 0
        self.steps_r = 0

        self.r = wheel_radius
        self.sep = wheel_sep

    # Retrieving step count from encoders. 
    # If encoder steps higher, then wheel has gone clockwise.
    # This is forward movement for right wheel and backwards for left wheel, 
    # so multiply by -1 for left.
    def get_steps(self, encoder, side):
        if (side == 0):     # zero for left side
            steps_curr = encoder.steps - self.steps_l
            self.steps_l = encoder.steps
            return (steps_curr * -1)
        elif (side == 1):
            steps_curr = encoder.steps - self.steps_r
            self.steps_r = encoder.steps
            return steps_curr

    # Motor encoder measurement which measures how fast wheel is turning
    # returns angular velocity of wheel
    def wheel_speed(self, steps):
        N = 32  # 32 line encoder
        GEAR_RATIO = 114.7
        rotations = steps/N
        rps_encoder = rotations/self.dt
        rps_wheel = rps_encoder / GEAR_RATIO
        w_angular_vel = rps_wheel * 2 * np.pi

        return w_angular_vel

    # Veclocity motion model
    def base_velocity(self, wl, wr):
        v = (wl * self.r + wr * self.r) / 2.0

        w = (wl - wr) / self.sep

        return v, w

    # Kinematic motion model
    def pose_update(self):
        steps_curr_l = self.get_steps(self, encoder_l, 0)
        steps_curr_r = self.get_steps(self, encoder_r, 1)

        self.wl = self.wheel_speed(steps_curr_l)
        self.wr = self.wheel_speed(steps_curr_r)

        v, w = self.base_velocity(self.wl, self.wr)

        self.x = self.x + self.dt * v * np.cos(self.th)
        self.y = self.y + self.dt * v * np.sin(self.th)
        self.th = self.th + w * self.dt
        self.th_deg = self.th * 180/np.pi

        return self.x, self.y, self.th


# Kinematic motion control
class RobotController:

    def __init__(self, Kp=0.1, Ki=0.01, wheel_radius=0.028, wheel_sep=0.101):
        self.Kp = Kp
        self.Ki = Ki
        self.r = wheel_radius
        self.l = wheel_sep
        self.e_sum_l = 0
        self.e_sum_r = 0

    def p_control(self, w_desired, w_measured, e_sum):
        duty_cycle = min(max(-1, self.Kp * (w_desired - w_measured) + self.Ki * e_sum), 1)

        e_sum = e_sum + (w_desired - w_measured)

        return duty_cycle, e_sum

    def drive(self, v_desired, w_desired, wl, wr):
        wl_desired = v_desired / self.r + self.l * w_desired / 2
        wr_desired = v_desired / self.r - self.l * w_desired / 2

        duty_cycle_l, self.e_sum_l = self.p_control(wl_desired, wl, self.e_sum_l)
        duty_cycle_r, self.e_sum_r = self.p_control(wr_desired, wr, self.e_sum_r)

        return duty_cycle_l, duty_cycle_r


class TentaclePlanner:

    def __init__(self, obstacles, dt=0.1, steps=5, alpha=1, beta=0.1):

        self.dt = dt
        self.steps = steps
        # Tentacles are possible trajectories to follow
        self.tentacles = [(0.0, 1.0), (0.0, -1.0), (0.1, 1.0), (0.1, -1.0), (0.1, 0.5), (0.1, -0.5), (0.1, 0.0),
                          (0.0, 0.0)]

        self.alpha = alpha
        self.beta = beta

        self.obstacles = obstacles

    # Play a trajectory and evaluate where you'd end up
    def roll_out(self, v, w, goal_x, goal_y, goal_th, x, y, th, sensor):

        for j in range(self.steps):

            x = x + self.dt * v * np.cos(th)
            y = y + self.dt * v * np.sin(th)
            th = (th + w * self.dt)

            if (self.check_collision(sensor)):
                return np.inf

        # Wrap angle error -pi,pi
        e_th = goal_th - th
        e_th = np.arctan2(np.sin(e_th), np.cos(e_th))

        cost = self.alpha * ((goal_x - x) ** 2 + (goal_y - y) ** 2) + self.beta * (e_th ** 2)

        return cost

    def check_collision(self, sensor):

        # Add ultrasonic sensors to detect obstacles

        min_dist = sensor.distance

        if (min_dist < 0.1):
            return True
        return False

    # Choose trajectory that will get you closest to the goal
    def plan(self, goal_x, goal_y, goal_th, x, y, th, sensor):

        costs = []
        for v, w in self.tentacles:
            costs.append(self.roll_out(v, w, goal_x, goal_y, goal_th, x, y, th, sensor))

        best_idx = np.argmin(costs)

        return self.tentacles[best_idx]

#wheel radius = 28mm
#wheel sep = 101mm

robot = DiffDriveRobot(inertia=5, dt=0.1, drag=1, wheel_radius=0.028, wheel_sep=0.101)
controller = RobotController(Kp=1,Ki=0.25,wheel_radius=0.028,wheel_sep=0.101)
planner = TentaclePlanner(dt=0.1,steps=5,alpha=1,beta=1e-5)

#Outputting desired velocities to motors

v_goal = 0.5 #target body velocity
w_goal = 1 #target angular velocity
duty_cycle_l,duty_cycle_r = controller.drive(v_goal,w_goal,robot.wl,robot.wr)
pwm_l.value = duty_cycle_l
pwm_r.value = duty_cycle_r

#Outputting desired velocities to motors with planning for goal trajectory

goal_x = 0.10  #x target
goal_y = 0.10  #y target
goal_th = 0 #orientation target

v_goal, w_goal = planner.plan(goal_x,goal_y,goal_th,robot.x,robot.y,robot.th)
duty_cycle_l,duty_cycle_r = controller.drive(v_goal,w_goal,robot.wl,robot.wr)
pwm_l.value = duty_cycle_l
pwm_r.value = duty_cycle_r

poses = []
velocities = []
duty_cycle_commands = []

goal_x = 0.10  # x target
goal_y = 0.10  # y target
goal_th = 0  # orientation target

for i in range(200):
    v_goal, w_goal = planner.plan(goal_x, goal_y, goal_th, robot.x, robot.y, robot.th, sensor_f)
    duty_cycle_l, duty_cycle_r = controller.drive(v_goal, w_goal, robot.wl, robot.wr)

    # Simulate robot motion - send duty cycle command to controller
    x, y, th = robot.pose_update(duty_cycle_l, duty_cycle_r)

    # Log data
    poses.append([x, y, th])
    duty_cycle_commands.append([duty_cycle_l, duty_cycle_r])
    velocities.append([robot.wl, robot.wr])