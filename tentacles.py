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
planner = TentaclePlanner(obstacle, dt=0.1,steps=5,alpha=1,beta=0.15)    # for prelim comp, don't care about orientation. so could change beta to 0 and remove goal_th

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
