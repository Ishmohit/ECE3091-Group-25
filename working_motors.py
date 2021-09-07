#%%

import numpy as np
from matplotlib import pyplot as plt
from IPython import display

class DiffDriveRobot:
    
    def __init__(self, dt=0.1, wheel_radius=0.028, wheel_sep=0.101):
        
        self.x = 0.0 # y-position
        self.y = 0.0 # y-position 
        self.th = 0.0 # orientation
        
        self.wl = 0.0 #rotational velocity left wheel
        self.wr = 0.0 #rotational velocity right wheel
        
        self.prevduty_l = 0 # prev right duty cycle
        self.prevduty_r = 0 # prev left duty cycle
        self.dt = dt
        
        self.r = wheel_radius
        self.l = wheel_sep
    
    # Calculate motor angular velocity from shaft encoder feedback
    def wheel_speed(self, steps, direc):

        N = 128 # Encoder state transitions
        RATIO = 114.7 # Gear ratio
        
        steps_ps = steps/self.dt
        rps_encoder = steps_ps/N
        rps_wheel = rps_encoder/RATIO
        w = 2*np.pi*rps_wheel
        if (direc == 1) & (steps < 200): 
            w = -w
        
        return w
    
    # Veclocity motion model
    def base_velocity(self,wl,wr):
        
        v = (wl*self.r + wr*self.r)/2.0
        
        w = (wl - wr)/self.l
        
        return v, w
    
    # Kinematic motion model
    def pose_update(self, duty_cycle_l, duty_cycle_r, dir_l, dir_r):

        # Get current steps
        steps_curr_l = (duty_cycle_l - 0.965*(duty_cycle_l - self.prevduty_l)) * 3300
        steps_curr_r = (duty_cycle_r - 0.965*(duty_cycle_r - self.prevduty_r)) * 3300

        # store duty cycle, for step estimation
        self.prevduty_l = (duty_cycle_l - 0.965*(duty_cycle_l - self.prevduty_l))
        self.prevduty_r = (duty_cycle_r - 0.965*(duty_cycle_r - self.prevduty_r))
        
        # Update angular velocities
        self.wl = self.wheel_speed(steps_curr_l, dir_l)
        self.wr = self.wheel_speed(steps_curr_r, dir_r)
        
        # Update robot position
        v, w = self.base_velocity(self.wl,self.wr)
        self.x = self.x + self.dt*v*np.cos(self.th)
        self.y = self.y + self.dt*v*np.sin(self.th)
        self.th = self.th + w*self.dt
        
        
        return self.x, self.y, self.th
        
class RobotController:
    
    def __init__(self,Kp=0.1,Ki=0.01,wheel_radius=0.028, wheel_sep=0.101):
        
        self.Kp = Kp
        self.Ki = Ki
        self.r = wheel_radius
        self.l = wheel_sep
        self.e_sum_l = 0
        self.e_sum_r = 0
        
    def p_control(self,w_desired,w_measured,e_sum):
        
        duty_cycle = min(max(-1,self.Kp*(w_desired-w_measured) + self.Ki*e_sum),1)
        
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
        
        if duty_cycle_l >= 0:
            dir_l = 0
        else: dir_l = 1
        if duty_cycle_r >= 0:
            dir_r = 0
        else: dir_r = 1  
        
        print('Dir_l:',dir_l)
        print('Dir_r:',dir_r)
            
        duty_cycle_l = abs(duty_cycle_l)
        duty_cycle_r = abs(duty_cycle_r)
        
        return duty_cycle_l, duty_cycle_r, dir_l, dir_r

robot = DiffDriveRobot(dt=0.1, wheel_radius=0.028, wheel_sep=0.101)
controller = RobotController(Kp=3,Ki=0.1,wheel_radius=0.028, wheel_sep=0.101)

plt.figure(figsize=(15,9))

poses = []
velocities = []
duty_cycle_commands = []
for i in range(300):

    # Example motion using controller 
    
    if i < 100: # drive in circular path (turn left) for 10 s
        duty_cycle_l,duty_cycle_r,dir_l, dir_r = controller.drive(0.1,1,robot.wl,robot.wr)
    elif i < 200: # drive in circular path (turn right) for 10 s
        duty_cycle_l,duty_cycle_r,dir_l, dir_r = controller.drive(0.1,-1,robot.wl,robot.wr)
    else: # stop
        duty_cycle_l,duty_cycle_r = (0,0)
    
    # Simulate robot motion - send duty cycle command to robot
    x,y,th = robot.pose_update(duty_cycle_l,duty_cycle_r, dir_l, dir_r)
    
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
# %%
