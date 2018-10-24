import numpy as np


class Env():

    def __init__(self):
        self.size_x = 21
        self.size_y = 25
        self.square_size = 5
        self.start_x = 135
        self.start_y = -9375
        self.goal_start = [55, -9480]
        self.goal_end = [74, -9440]
        self.end_x = self.start_x - self.size_x * self.square_size
        self.end_y = self.start_y - self.size_y * self.square_size
        self.current_x = self.start_x
        self.current_y = self.start_y
        self.init_state = 0
        self.avg = self.__init_avg()
        self.grid = self.__init_grid()
        self.init_x = 0
        self.init_y = 0
        self.objects = self.__init_objects()
        self.objects_circles = self.__init_objects_circles()
        self.grid = self.__init_grid()

    def __init_objects(self):

        obj_1 = [[132.7, -9389], [120.8, -9399]]
        obj_2 = [[119, -9418.8], [110, -9426]]
        obj_3 = [[110.8, -9418.8], [117.9, -9426.3]]
        obj_4 = [[104, -9410], [98.5, -9414]]
        obj_5 = [[102.46, -9404], [96.3, -9416.6]]
        obj_6 = [[90.4, -9394], [81.5, -9407.3]]
        obj_7 = [[87.1, -9389.4], [73, -9393.7]]
        obj_8 = [[71.5, -9422.7], [31.9, -9407.2]]
        obj_9 = [[143.1, -9454.4], [130.8, -9468.2]]
        obj_10 = [[134.1, -9472.6], [118.7, -9475.3]]
        obj_11 = [[108, -9471.3], [74.2, -9447.2]]
        obj_12 = [[129.4, -9446.9], [120.5, -9441.9]]
        obj_13 = [[116.3, -9430.2], [108.2, -9437.2]]
        '''
        obj_1 = [[101, -9436.6][121.15, -9437.5]]
        #smedrectangle
        101
        -9436.6
        121.15
        -9437.5
        '''
        objects = [
                    obj_1, 
                    obj_2, 
                    obj_3, 
                    obj_4, 
                    obj_5, 
                    obj_6, 
                    obj_7, 
                    obj_8, 
                    obj_8, 
                    obj_9, 
                    obj_10, 
                    obj_11, 
                    obj_12, 
                    obj_13
                    ]


        return objects

    def __init_objects_circles(self):
        #circle
        #110.23
        #-9447.7
        #radius = (-9447) - (-9460.2)

        obj_1 = [[110.23, -9447.7], [13.2**2]]

        objects = [obj_1]
        return np.array(objects)

    def __init_grid(self):
        grid = np.zeros(self.size_y, self.size_x)

        state = 0
        x = self.init_x
        y = self.init_y

        for i in range(self.size_y):
           
            x = self.init_x
            
            for j in range(self.size_y):

                up = -1 if i == 0 else state - self.size_x
                down = -1 if i == self.size_y - 1 else state + self.size_x
                left = -1 if j == 0 else state - 1
                right = -1 if j == self.size_x - 1 else state + 1

                neighbours = [up, down, left, right]
                grid[i, j] = (state, neighbours, x, y)
                
                x -= self.square_size

                if state == self.init_state: 
                    current_y = i
                    current_x = j

                    self.init_x = current_x
                    self.init_y = current_y
			
                state += 1
		
            y -= self.square_size
	
        return grid

    def __init_avg(self):

        avg_x = (self.goal_start[0] + self.goal_end[0]) / 2
        avg_y = (self.goal_start[1] + self.goal_end[1]) / 2
        
        return avg_x, avg_y
        
    def is_goal(self, x, y):
        
        within_x = (x >= self.goal_start[0]) and (x <= self.goal_end[0])
        within_y = (y >= self.goal_start[1]) and (y <= self.goal_end[1])

        return within_x and within_y

    def is_within_bounds(self, x, y):

        if self.is_within_circle(x, y): return True

        for obj in self.objects:
            within_x = (x >= obj[0, 0]) and (x <= obj[1, 0])
            within_y = (y >= obj[0, 1]) and (y <= obj[1, 1])
            
            if within_x and within_y: 
                return True

        return False

    def is_within_circle(self, x, y):
        
        #d=(xp−xc)2+(yp−yc)2−−−−−−−−−−−−−−−−−−√.
        
        for _, obj in enumerate(self.objects_circles):
            dist = (x - obj[0, 0])**2 + (y - obj[0, 1])**2
            if dist <= obj[1, 0]: return True

        return False

    def step(self, action):
        
        next_state = self.grid.neighbours[action]

        x = self.grid.current_x
        y = self.grid.current_y

        dist = self.distance(x, y)

        terminal = False
        reward = 0

        if self.is_goal(x, y):
            terminal = True
            reward = 100

        elif next_state != -1:
            
            stuck = self.is_within_bounds(x, y)
            
            if stuck:
                reward = -100 - (dist * dist * 10)
                terminal = True

        else :
            next_state = current_state
            reward = -100 * 100000

        goal = self.is_goal()

        next_state = [new_x, new_y, x_diff, y_diff, hp]

        return next_state, reward, terminal

    def reset(self):
        return self.start_x, self.start_y

    def distance(self, x, y):

        avg_x, avg_y = self.avg 
        
        a = x - avg_x
        b = y - avg_y

        return np.sqrt( a**2 + b**2 )