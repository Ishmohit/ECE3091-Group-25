import numpy as np
from matplotlib import pyplot as plt
from IPython import display
import gpiozero
import time

class DiffDriveRobot:
    
    def __init__(self, dt=0.1, wheel_radius=0.028, wheel_sep=0.101):
        
        self.x = 0.0 # y-position
        self.y = 0.0 # y-position 
        self.th = 0.0 # orientation
        
        self.wl = 0.0 #rotational velocity left wheel
        self.wr = 0.0 #rotational velocity right wheel

        self.prevsteps_l = 0 # previous right encoder steps
        self.prevsteps_r = 0 # previous left encoder steps

        self.dt = dt
        
        self.r = wheel_radius
        self.l = wheel_sep

        #pins
        self.pwm_l = gpiozero.PWMOutputDevice(pin=18,active_high=True,initial_value=0,frequency=50000)
        self.dir_l = gpiozero.OutputDevice(pin=23)

        self.pwm_r = gpiozero.PWMOutputDevice(pin=15,active_high=True,initial_value=0,frequency=50000)
        self.dir_r = gpiozero.OutputDevice(pin=14)

        self.encoder_l = gpiozero.RotaryEncoder(a=5, b=6,max_steps=100000) 
        self.encoder_r = gpiozero.RotaryEncoder(a=25, b=8,max_steps=100000) 

    def get_steps(self, encoder, side): # might need to multiply one side by -1, to account for one going cw and the other ccw?
        if (side == 0):
            steps_curr = encoder.steps - self.prevsteps_l
            self.steps_l = encoder.steps
            return (steps_curr)
        elif (side == 1):
            steps_curr = encoder.steps - self.prevsteps_r
            self.steps_r = encoder.steps
            return steps_curr
    
    # Calculate motor angular velocity from shaft encoder feedback
    def wheel_speed(self, steps):

        N = 128 # Encoder state transitions
        RATIO = 114.7 # Gear ratio
        
        steps_ps = steps/self.dt
        rps_encoder = steps_ps/N
        rps_wheel = rps_encoder/RATIO
        w = 2*np.pi*rps_wheel
        
        return w
    
    # Veclocity motion model
    def base_velocity(self,wl,wr):
        
        v = (wl*self.r + wr*self.r)/2.0
        
        w = (wl - wr)/self.l
        
        return v, w
    
    # Kinematic motion model
    def pose_update(self):

        # Get current steps
        steps_curr_l = self.get_steps(self.encoder_l, 0)
        steps_curr_r = self.get_steps(self.encoder_r, 1)
        
        # Update angular velocities
        self.wl = self.wheel_speed(steps_curr_l)
        self.wr = self.wheel_speed(steps_curr_r)

        # Update robot position
        v, w = self.base_velocity(self.wl,self.wr)
        self.x = self.x + self.dt*v*np.cos(self.th)
        self.y = self.y + self.dt*v*np.sin(self.th)
        self.th = self.th + w*self.dt
        
        return self.x, self.y, self.th

    def motor_update(self):
        # Update motor output
        if duty_cycle_l >=0:
            self.dir_l.value = 0
        else: self.dir_l.value = 1
        if duty_cycle_r >=0:
            self.dir_r.value = 0
        else: self.dir_r.value = 1

        self.pwm_l.value = abs(duty_cycle_l)
        self.pwm_r.value = abs(duty_cycle_r)
        
class RobotController:
    
    def __init__(self,Kp=0.1,Ki=0.01,wheel_radius=0.028, wheel_sep=0.101):
        
        self.Kp = Kp
        self.Ki = Ki
        self.r = wheel_radius
        self.l = wheel_sep
        self.e_sum_l = 0
        self.e_sum_r = 0
        
    def p_control(self,w_desired,w_measured,e_sum):
        
        duty_cycle = min(max(-1,self.Kp*(w_desired-w_measured) + self.Ki*e_sum),1) # if step counts are incorrect, maybe lower the max duty cycle from one to like 0.5
        
        e_sum = e_sum + (w_desired-w_measured)
        
        return duty_cycle, e_sum
        
        
    # Calculate duty cycles from target linear and angular velocity 
    def drive(self,v_desired,w_desired,wl,wr):
        
        wl_desired = v_desired/self.r + self.l*w_desired/2 
        wr_desired = v_desired/self.r - self.l*w_desired/2
        print('Wl_d:',wl_desired)
        print('Wr_d:',wr_desired)
        
        duty_cycle_l,self.e_sum_l = self.p_control(wl_desired,wl,self.e_sum_l)
        duty_cycle_r,self.e_sum_r = self.p_control(wr_desired,wr,self.e_sum_r)
        
        return duty_cycle_l, duty_cycle_r

robot = DiffDriveRobot(dt=0.1, wheel_radius=0.028, wheel_sep=0.101)
controller = RobotController(Kp=3,Ki=0.1,wheel_radius=0.028, wheel_sep=0.101)                # CALIBRATE KP AND KI

plt.figure(figsize=(15,9))

poses = []
velocities = []
duty_cycle_commands = []
for i in range(300):

    # Example motion using controller 
    
    if i < 100: # drive in circular path (turn left) for 10 s
        duty_cycle_l,duty_cycle_r = controller.drive(0.1,1,robot.wl,robot.wr)
    elif i < 200: # drive in circular path (turn right) for 10 s
        duty_cycle_l,duty_cycle_r = controller.drive(0.1,-1,robot.wl,robot.wr)
    else: # stop
        duty_cycle_l,duty_cycle_r = (0,0)
    
    # Simulate robot motion - send duty cycle command to robot
    x,y,th = robot.pose_update()
    robot.motor_update(duty_cycle_l,duty_cycle_r)
    
    # Log data
    poses.append([x,y,th])
    duty_cycle_commands.append([duty_cycle_l,duty_cycle_r])
    velocities.append([robot.wl,robot.wr])
    
    # Plot robot data
    plt.clf()
    plt.cla()
    plt.subplot(1,2,1)
    plt.plot(np.array(poses)[:,0],np.array(poses)[:,1])
    plt.plot(x,y,'k',marker='+')
    plt.quiver(x,y,0.1*np.cos(th),0.1*np.sin(th))
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel('x-position (m)')
    plt.ylabel('y-position (m)')
    plt.grid()
    
    plt.subplot(2,2,2)
    plt.plot(np.arange(i+1)*robot.dt,np.array(duty_cycle_commands))
    plt.xlabel('Time (s)')
    plt.ylabel('Duty cycle')
    plt.grid()
    
    plt.subplot(2,2,4)
    plt.plot(np.arange(i+1)*robot.dt,np.array(velocities))
    plt.xlabel('Time (s)')
    plt.ylabel('Wheel $\omega$')
    plt.legend(['Left wheel', 'Right wheel'])
    plt.grid()
    
    
    display.clear_output(wait=True)
    display.display(plt.gcf())