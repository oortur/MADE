from tkinter import *
import math
import heapq
from copy import copy
from shapely.geometry import Point, Polygon

'''================= Your classes and methods ================='''

# These functions will help you to check collisions with obstacles

def rotate(points, angle, center):
    angle = math.radians(angle)
    cos_val = math.cos(angle)
    sin_val = math.sin(angle)
    cx, cy = center
    new_points = []

    for x_old, y_old in points:
        x_old -= cx
        y_old -= cy
        x_new = x_old * cos_val - y_old * sin_val
        y_new = x_old * sin_val + y_old * cos_val
        new_points.append([x_new+cx, y_new+cy])

    return new_points

def get_polygon_from_position(position) :
    x,y,yaw = position
    points = [(x - 50, y - 100), (x + 50, y - 100), (x + 50, y + 100), (x - 50, y + 100)] 
    new_points = rotate(points, yaw * 180 / math.pi, (x,y))
    return Polygon(new_points)

def get_polygon_from_obstacle(obstacle) :
    points = [[obstacle[0], obstacle[1]], [obstacle[2], obstacle[3]], [obstacle[4], obstacle[5]], [obstacle[6], obstacle[7]]]
    return Polygon(points)

def collides(position, obstacle) :
    return get_polygon_from_position(position).intersection(get_polygon_from_obstacle(obstacle))


class State:
    """
    Class for State object to store coordinates and yaw of this state.
    Extra attributes are:
    	actions_history - stores list of actions from start to this state
    	step_distance - distance covered by car on one step (you may change this parameter for faster/slower convergence)
    """
    def __init__(self, x, y, yaw, actions_history=None, step_distance=10):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.actions_history = actions_history if actions_history is not None else []
        self.step_distance = step_distance

    def move(self, action):
        """
        Returns new state under given action.

        I consider three possible actions:
            ( 0): move FORWARD (on predefined distance self.step_distance) 
            ( 1): move RIGHT (change yaw, then move forward on half of self.step_distance)
            (-1): move LEFT (same as RIGHT with another turn)
        """
        if action == 0:
            # FORWARD
            x = self.x + self.step_distance * math.sin(self.yaw)
            y = self.y - self.step_distance * math.cos(self.yaw)
            yaw = self.yaw
        else:
            # RIGHT or LEFT
            yaw = self.yaw + action * math.pi * 0.1
            if yaw > math.pi:
                yaw -= 2 * math.pi
            if yaw <= -math.pi:
                yaw += 2 * math.pi
            x = self.x + self.step_distance / 2 * math.sin(yaw)
            y = self.y - self.step_distance / 2 * math.cos(yaw)
        actions_history = copy(self.actions_history) + [action]
        return State(x, y, yaw, actions_history)

    def __lt__(self, other):
        # needed to compare states in queue
        return True


class Window():
        
    '''================= Your Main Function ================='''
    
    def check_collisions(self, state):
        """returns True if NO collisions happened"""
        for obs in self.get_obstacles():
            if collides((state.x, state.y, state.yaw), obs):
                return False
        return True


    def target_status(self, state, target, dist_delta=3):
        """returns True if target was found"""
        if self.distance(state.x, state.y, target.x, target.y) <= dist_delta:
            angle_diff = abs(state.yaw - target.yaw)
            angle_diff = min(angle_diff, 2 * math.pi - angle_diff) / math.pi
            if angle_diff <= 0.1:
                return True
        return False


    def get_next_states(self, state):
        """returns list of possible next states"""
        next_states = []
        for action in [-1, 0, 1]:
            new_state = state.move(action)
            if self.check_collisions(new_state):
                next_states.append(new_state)
        return next_states


    def cosine(self, x1, y1, x2, y2):
        """returns cosine of angle between two vectors"""
        dot = x1 * x2 + y1 * y2
        d1 = self.distance(0, 0, x1, y1)
        d2 = self.distance(0, 0, x2, y2)
        if d1 * d2 == 0:
            return 0
        return dot / d1 / d2


    def codirectness(self, x1, y1, x2, y2):
        """
        Measure of codirectness in [0, 1]
        0 - vectors are codirected (angle equals zero)
        1 - vectors are collinear but not codirected (angle equals pi)
        """
        return (1 - self.cosine(x1, y1, x2, y2)) * 0.5


    def heuristic(self, state, target):
        """
        Heuristic for closeness of current state and target.

        We calculate pure distance from state to target (assuming no obstacles are on the field),
        and codirectness of three vectors:
            - vector responding to current state yaw
            - vector responding to target state yaw
            - vector respondong to direction from state to target

        The heuristic minimizes product of distance (main value to minimize) and weighted sum of codirectnesses
        """
        dist_to_target = self.distance(state.x, state.y, target.x, target.y)

        state_dist_codirectness = self.codirectness(math.sin(state.yaw), -math.cos(state.yaw), target.x - state.x, target.y - state.y)
        target_dist_codirectness = self.codirectness(math.sin(target.yaw), -math.cos(target.yaw), target.x - state.x, target.y - state.y)
        state_target_codirectness = self.codirectness(math.sin(state.yaw), -math.cos(state.yaw), math.sin(target.yaw), -math.cos(target.yaw))
        # every codirectness metric in [0, 1]

        return dist_to_target * (5 + 10 * state_dist_codirectness + 1 * target_dist_codirectness + 1 * state_target_codirectness)


    def a_star_search(self, start, target):
        """A* algorithm to search shortest path."""
        queue = []
        heapq.heappush(queue, (self.heuristic(start, target), 0, start))
        visited = set()
        step = 0
        while len(queue) > 0:
            if step > 1e6:
                break
            state = heapq.heappop(queue)[2]

            if self.target_status(state, target):
                print(f"PATH FOUND\nSteps: {step}\n")
                return state

            round_state = (round(state.x), round(state.y), round(state.yaw, 2))
            if round_state in visited:
                continue
            visited.add(round_state)
            for new_state in self.get_next_states(state):
                if (round(new_state.x), round(new_state.y), round(new_state.yaw, 2)) in visited:
                    continue
                heapq.heappush(queue, (self.heuristic(new_state, target), step, new_state))
            step += 1
        return None


    # on click GO
    def go(self, event):

        # print("Start position:", self.get_start_position())
        # print("Target position:", self.get_target_position()) 
        # print("Obstacles:", self.get_obstacles())

        start = State(*self.get_start_position())
        state = State(*self.get_start_position())
        target = State(*self.get_target_position())

        final_state = self.a_star_search(state, target)
        if final_state is None:
            print("PATH NOT FOUND\n")
            return

        path = [state.x, state.y]
        actions = final_state.actions_history
        for action in actions:
            state = state.move(action)
            path += [state.x, state.y]
        self.canvas.create_line(*path, dash=(1,1), fill='cyan', arrow=LAST)

        
    '''================= Interface Methods ================='''
    
    def get_obstacles(self) :
        obstacles = []
        potential_obstacles = self.canvas.find_all()
        for i in potential_obstacles:
            if (i > 2) :
                coords = self.canvas.coords(i)
                if coords:
                    obstacles.append(coords)
        return obstacles
            
            
    def get_start_position(self) :
        x,y = self.get_center(2) # Purple block has id 2
        yaw = self.get_yaw(2)
        return x,y,yaw
    
    def get_target_position(self) :
        x,y = self.get_center(1) # Green block has id 1 
        yaw = self.get_yaw(1)
        return x,y,yaw 
 

    def get_center(self, id_block):
        coords = self.canvas.coords(id_block)
        center_x, center_y = ((coords[0] + coords[4]) / 2, (coords[1] + coords[5]) / 2)
        return [center_x, center_y]

    def get_yaw(self, id_block):
        center_x, center_y = self.get_center(id_block)
        first_x = 0.0
        first_y = -1.0
        second_x = 1.0
        second_y = 0.0
        points = self.canvas.coords(id_block)
        end_x = (points[0] + points[2])/2
        end_y = (points[1] + points[3])/2
        direction_x = end_x - center_x
        direction_y = end_y - center_y
        length = math.hypot(direction_x, direction_y)
        unit_x = direction_x / length
        unit_y = direction_y / length
        cos_yaw = unit_x * first_x + unit_y * first_y 
        sign_yaw = unit_x * second_x + unit_y * second_y
        if (sign_yaw >= 0 ) :
            return math.acos(cos_yaw)
        else :
            return -math.acos(cos_yaw)
       
    def get_vertices(self, id_block):
        return self.canvas.coords(id_block)

    '''=================================================='''

    def rotate(self, points, angle, center):
        angle = math.radians(angle)
        cos_val = math.cos(angle)
        sin_val = math.sin(angle)
        cx, cy = center
        new_points = []

        for x_old, y_old in points:
            x_old -= cx
            y_old -= cy
            x_new = x_old * cos_val - y_old * sin_val
            y_new = x_old * sin_val + y_old * cos_val
            new_points.append(x_new+cx)
            new_points.append(y_new+cy)

        return new_points

    def start_block(self, event):
        widget = event.widget
        widget.start_x = event.x
        widget.start_y = event.y

    def in_rect(self, point, rect):
        x_start, x_end = min(rect[::2]), max(rect[::2])
        y_start, y_end = min(rect[1::2]), max(rect[1::2])

        if x_start < point[0] < x_end and y_start < point[1] < y_end:
            return True

    def motion_block(self, event):
        widget = event.widget

        for i in range(1, 10):
            if widget.coords(i) == []:
                break
            if self.in_rect([event.x, event.y], widget.coords(i)):
                coords = widget.coords(i)
                id = i
                break

        res_cords = []
        try:
            coords
        except:
            return

        for ii, i in enumerate(coords):
            if ii % 2 == 0:
                res_cords.append(i + event.x - widget.start_x)
            else:
                res_cords.append(i + event.y - widget.start_y)

        widget.start_x = event.x
        widget.start_y = event.y
        widget.coords(id, res_cords)
        widget.center = ((res_cords[0] + res_cords[4]) / 2, (res_cords[1] + res_cords[5]) / 2)

    def draw_block(self, points, color):
        x = self.canvas.create_polygon(points, fill=color)
        return x

    def distance(self, x1, y1, x2, y2):
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def set_id_block(self, event):
        widget = event.widget

        for i in range(1, 10):
            if widget.coords(i) == []:
                break
            if self.in_rect([event.x, event.y], widget.coords(i)):
                coords = widget.coords(i)
                id = i
                widget.id_block = i
                break

        widget.center = ((coords[0] + coords[4]) / 2, (coords[1] + coords[5]) / 2)

    def rotate_block(self, event):
        angle = 0
        widget = event.widget

        if widget.id_block == None:
            for i in range(1, 10):
                if widget.coords(i) == []:
                    break
                if self.in_rect([event.x, event.y], widget.coords(i)):
                    coords = widget.coords(i)
                    id = i
                    widget.id_block == i
                    break
        else:
            id = widget.id_block
            coords = widget.coords(id)

        wx, wy = event.x_root, event.y_root
        try:
            coords
        except:
            return

        block = coords
        center = widget.center
        x, y = block[2], block[3]

        cat1 = self.distance(x, y, block[4], block[5])
        cat2 = self.distance(wx, wy, block[4], block[5])
        hyp = self.distance(x, y, wx, wy)

        if wx - x > 0: angle = math.acos((cat1**2 + cat2**2 - hyp**2) / (2 * cat1 * cat2))
        elif wx - x < 0: angle = -math.acos((cat1**2 + cat2**2 - hyp**2) / (2 * cat1 * cat2))

        new_block = self.rotate([block[0:2], block[2:4], block[4:6], block[6:8]], angle, center)
        self.canvas.coords(id, new_block)

    def delete_block(self, event):
        widget = event.widget.children["!canvas"]

        for i in range(1, 10):
            if widget.coords(i) == []:
                break
            if self.in_rect([event.x, event.y], widget.coords(i)):
                widget.coords(i, [0,0])
                break

    def create_block(self, event):
        block = [[0, 100], [100, 100], [100, 300], [0, 300]]

        id = self.draw_block(block, "black")

        self.canvas.tag_bind(id, "<Button-1>", self.start_block)
        self.canvas.tag_bind(id, "<Button-3>", self.set_id_block)
        self.canvas.tag_bind(id, "<B1-Motion>", self.motion_block)
        self.canvas.tag_bind(id, "<B3-Motion>", self.rotate_block)

    def make_draggable(self, widget):
        widget.bind("<Button-1>", self.drag_start)
        widget.bind("<B1-Motion>", self.drag_motion)

    def drag_start(self, event):
        widget = event.widget
        widget.start_x = event.x
        widget.start_y = event.y

    def drag_motion(self, event):
        widget = event.widget
        x = widget.winfo_x() - widget.start_x + event.x + 200
        y = widget.winfo_y() - widget.start_y + event.y + 100
        widget.place(rely=0.0, relx=0.0, x=x, y=y)

    def create_button_create(self):
        button = Button(
            text="New",
            bg="#555555",
            activebackground="blue",
            borderwidth=0
        )

        button.place(rely=0.0, relx=0.0, x=200, y=100, anchor=SE, width=200, height=100)
        button.bind("<Button-1>", self.create_block)

    def create_green_block(self, center_x):
        block = [[center_x - 50, 100],
                 [center_x + 50, 100],
                 [center_x + 50, 300],
                 [center_x - 50, 300]]

        id = self.draw_block(block, "green")

        self.canvas.tag_bind(id, "<Button-1>", self.start_block)
        self.canvas.tag_bind(id, "<Button-3>", self.set_id_block)
        self.canvas.tag_bind(id, "<B1-Motion>", self.motion_block)
        self.canvas.tag_bind(id, "<B3-Motion>", self.rotate_block)

    def create_purple_block(self, center_x, center_y):
        block = [[center_x - 50, center_y - 300],
                 [center_x + 50, center_y - 300],
                 [center_x + 50, center_y - 100],
                 [center_x - 50, center_y - 100]]

        id = self.draw_block(block, "purple")

        self.canvas.tag_bind(id, "<Button-1>", self.start_block)
        self.canvas.tag_bind(id, "<Button-3>", self.set_id_block)
        self.canvas.tag_bind(id, "<B1-Motion>", self.motion_block)
        self.canvas.tag_bind(id, "<B3-Motion>", self.rotate_block)

    def create_button_go(self):
        button = Button(
            text="Go",
            bg="#555555",
            activebackground="blue",
            borderwidth=0
        )

        button.place(rely=0.0, relx=1.0, x=0, y=200, anchor=SE, width=100, height=200)
        button.bind("<Button-1>", self.go)

    def run(self):
        root = self.root

        self.create_button_create()
        self.create_button_go()
        self.create_green_block(self.width/2)
        self.create_purple_block(self.width/2, self.height)

        root.bind("<Delete>", self.delete_block)

        root.mainloop()
        
    def __init__(self):
        self.root = Tk()
        self.root.title("")
        self.width  = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()
        self.root.geometry(f'{self.width}x{self.height}')
        self.canvas = Canvas(self.root, bg="#777777", height=self.height, width=self.width)
        self.canvas.pack()
        # self.points = [0, 500, 500/2, 0, 500, 500]
    
if __name__ == "__main__":
    run = Window()
    run.run()
