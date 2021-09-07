#%%

import numpy as np
from matplotlib import pyplot as plt
from IPython import display

class DiffDriveRobot:
    
    def __init__(self, dt=0.1, wheel_radius=0.028, wheel_sep=0.101):
        
        self.x = 0.0 # y-position in m
        self.y = 0.0 # y-position in m
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
        if (direc == 1) & (steps < 400): 
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

class TentaclePlanner:
    
    def __init__(self,obstacles,dt=0.1,steps=5,alpha=1,beta=0.1):
        
        self.dt = dt
        self.steps = steps
        # Tentacles are possible trajectories to follow
        self.tentacles = [(0.0,1.0),(0.0,-1.0),(0.0,0.5),(0.0,-0.5),(0.1,1.0),(0.1,-1.0),(0.1,0.5),(0.1,-0.5),(0.1,0.0),(-0.1,0.0),(0.0,0.0)]
        
        self.alpha = alpha
        self.beta = beta
        
        self.obstacles = obstacles
    
    # Play a trajectory and evaluate where you'd end up
    def roll_out(self,v,w,goal_x,goal_y,goal_th,x,y,th):
        
        for j in range(self.steps):
        
            x = x + self.dt*v*np.cos(th)
            y = y + self.dt*v*np.sin(th)
            th = (th + w*self.dt)
            
            if (self.check_collision(x,y)):
                return np.inf
        
        # Wrap angle error -pi,pi
        e_th = goal_th-th
        e_th = np.arctan2(np.sin(e_th),np.cos(e_th))
        
        cost = self.alpha*((goal_x-x)**2 + (goal_y-y)**2) + self.beta*(e_th**2)
        
        return cost
    
    def check_collision(self,x,y):
        
        min_dist = np.min(np.sqrt((x-self.obstacles[0])**2+(y-self.obstacles[1])**2))
        
        if (min_dist < 0.1):
            return True
        return False
        
    
    # Choose trajectory that will get you closest to the goal
    def plan(self,goal_x,goal_y,goal_th,x,y,th):
        
        costs =[]
        for v,w in self.tentacles:
            costs.append(self.roll_out(v,w,goal_x,goal_y,goal_th,x,y,th))
        
        best_idx = np.argmin(costs)
        
        return self.tentacles[best_idx]

    def intermediates(self,goal_x,goal_y,goal_th,x,y, th):
        turn = 0
        if (np.sqrt((goal_x - x)**2+(goal_y-y)**2)) < 0.1:
            int_goal_x = goal_x
            int_goal_y = goal_y
            int_goal_th = goal_th
        else:
            int_goal_x = x + (goal_x - x)/2
            int_goal_y = y + (goal_y - y)/2
            int_goal_th = np.arctan2(goal_y - robot.y,goal_x - robot.x)
            if abs(int_goal_th - th) > 1:
                turn = 1
        return int_goal_x, int_goal_y, int_goal_th, turn

obstacle = np.array([0.15, 0.15])
robot = DiffDriveRobot(dt=0.1, wheel_radius=0.028, wheel_sep=0.101)
controller = RobotController(Kp=3,Ki=0.1,wheel_radius=0.028, wheel_sep=0.101)        
planner = TentaclePlanner(obstacle, dt=0.1,steps=5,alpha=1,beta=0.15)    # remember for prelim comp, don't care about orientation

plt.figure(figsize=(15,9))

poses = []
velocities = []
duty_cycle_commands = []

goal_x = 0.3
goal_y = 0.3
goal_th = np.pi/2

for i in range(200):
    # Plan using tentacles
    int_goal_x, int_goal_y, int_goal_th, turn = planner.intermediates(goal_x,goal_y,goal_th,robot.x,robot.y, robot.th)
    if turn:
        v, w = 0, int_goal_th
    else:
        v,w = planner.plan(int_goal_x, int_goal_y, int_goal_th,robot.x,robot.y,robot.th)
    
    duty_cycle_l,duty_cycle_r, dir_l, dir_r = controller.drive(v,w,robot.wl,robot.wr)
    
    # Simulate robot motion - send duty cycle command to robot
    x,y,th = robot.pose_update(duty_cycle_l,duty_cycle_r, dir_l, dir_r)
    
    # Log data
    poses.append([x,y,th])
    duty_cycle_commands.append([duty_cycle_l,duty_cycle_r])
    velocities.append([robot.wl,robot.wr])
    
    # Plot robot data
    plt.clf()
    plt.subplot(1,2,1)
    plt.plot(np.array(poses)[:,0],np.array(poses)[:,1])
    plt.plot(x,y,'k',marker='+')
    plt.quiver(x,y,0.1*np.cos(th),0.1*np.sin(th))
    plt.plot(goal_x,goal_y,'x',markersize=5)
    plt.quiver(goal_x,goal_y,0.1*np.cos(goal_th),0.1*np.sin(goal_th))
    plt.plot(int_goal_x,int_goal_y,'x',markersize=5)
    plt.quiver(int_goal_x,int_goal_y,0.1*np.cos(int_goal_th),0.1*np.sin(int_goal_th))
    
    plt.plot(obstacle[0],obstacle[1],'ko',markersize=15)
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
