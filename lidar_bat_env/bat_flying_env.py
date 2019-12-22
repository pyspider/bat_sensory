from collections import namedtuple
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from .lidar_bat import LidarBat

FPS = 60

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Segment(object):
    def __init__(self, p0: Point, p1: Point):
        self.p0 = p0
        self.p1 = p1
 
def cal_cross_point(s0: Segment, s1: Segment) -> Point:
    x0, y0, x1, y1 = s0.p0.x, s0.p0.y, s0.p1.x, s0.p1.y
    x2, y2, x3, y3 = s1.p0.x, s1.p0.y, s1.p1.x, s1.p1.y
    den = (x3 - x2) * (y1 - y0) - (x1 - x0) * (y3 - y2)
    if den == 0:
        return Point(np.inf, np.inf)
    
    d1 = (y2 * x3 - x2 * y3)
    d2 = (y0 * x1 - x0 * y1)

    x = (d1 * (x1 - x0) - d2 * (x3 - x2)) / den
    y = (d1 * (y1 - y0) - d2 * (y3 - y2)) / den
    return Point(x, y)

def is_point_in_segment(p: Point, s: Segment):
    e = 1e-8
    x_ok = min(s.p0.x, s.p1.x) - e <= p.x and p.x <= max(s.p0.x, s.p1.x) + e
    y_ok = min(s.p0.y, s.p1.y) - e <= p.y and p.y <= max(s.p0.y, s.p1.y) + e
    return x_ok and y_ok



class BatFlyingEnv(gym.Env):
    """
    Description:
        Bats emit a pulse and receive the echo to calculate the distance and
        the direction of a object. So, they can fly without bumping some
        obstacle, and forage in the dark.

        In this environment, an agent can get the distance and the direction of
        the nearest obstacle when emits a pulse.

    Observation:
        Type: Box(2)
        Num  Observation     Min      Max
        0    echo distance  0        Inf
        1    echo direction -180 deg 180 deg

    Actions:
        Type: Box(6)
        Num   Action
        0     Acceleration
        1     direction to accelerate
        4     Emit Pulse
        5     Pulse direction

    Reward:
        Reword is 1 for every step take, including the termination step

    Starting State:
        position
        direction
        speed

    """

    metadata = {
        'render.model': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(
            self,
            world_width,
            world_height,
            discrete_length,
            dt=0.005,
            bat=None,
            walls=None,
            goal_area=None,
            accel_thresh=None,
            angle_thresh_radians=None):
        self.world_width = world_width
        self.world_height = world_height
        self.discrete_length = discrete_length
        self.dt = 0.005  # [s]

        walls = [
            (0, 0, 0, world_height),  # x0, y0, x1, y1
            (0, world_height, world_width, world_height),
            (world_width, 0, world_width, world_height),
            (0, 0, world_width, 0)
        ]
        self.walls = [] if walls is None else walls

        self.goal_area = () if goal_area is None else goal_area
        self.accel_thresh = 50  # [m/s^2]
        self.angle_thresh_radians = math.pi / 2 # [rad]

        self.action_space = spaces.Box(
            np.array([
                -self.accel_thresh,
                -self.angle_thresh_radians,
                0,
                -self.angle_thresh_radians]),
            np.array([
                self.accel_thresh,
                self.angle_thresh_radians,
                1,
                self.angle_thresh_radians]),
            dtype=np.float32)
        
        self.observation_space = spaces.Box(
            np.zeros(2),
            np.array([np.inf, 1]),
            dtype=np.float32)
        
        self.bat = LidarBat(0, 0.3, 0.75, 3, self.dt) if bat is None else bat
        self.viewer = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
   
    def step(self, action):
        bat_p0 = Point(self.bat.x, self.bat.y)
        self.bat.move(action[0], action[1])
        bat_p1 = Point(self.bat.x, self.bat.y)
        bat_seg = Segment(bat_p0, bat_p1)
        for w in self.walls:
            w_p0, w_p1 = Point(w[0], w[1]), Point(w[2], w[3])
            wall_seg = Segment(w_p0, w_p1)
            c_p = cal_cross_point(bat_seg, wall_seg)
            if is_point_in_segment(c_p, bat_seg):
                wall_angle = math.atan2(w_p1.y - w_p0.y, w_p1.x - w_p0.x)
                self.bat.bump(bat_p0.x, bat_p0.y, wall_angle)
                step_reward = -1.0
        done = None
        step_reward = 0
        return np.array(self.bat.state), step_reward, done, {}

    def reset(self, bat=None):
        self.bat = self.bat if bat is None else bat
        self.reward = 0.0
        self.t = 0.0
        return np.array(self.bat.state)

    def render(self, screen_width=600, mode='human'):
        aspect_ratio = self.world_height / self.world_width
        screen_height = int(aspect_ratio * screen_width)
        scale = screen_width / self.world_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            r = (self.bat.size * scale) / 2
            wing = 4 * math.pi / 5 # angle [rad]
            nose_x, nose_y = r, 0
            r_x, r_y = r * math.cos(-wing), r * math.sin(-wing)
            l_x, l_y = r * math.cos(+wing), r * math.sin(+wing)
            bat_geom = rendering.FilledPolygon([
                (nose_x, nose_y),
                (r_x, r_y),
                (l_x, l_y)])
            bat_geom.set_color(0, 0, 0)
            self.battrans = rendering.Transform()
            bat_geom.add_attr(self.battrans)
            self.viewer.add_geom(bat_geom)
            self._bat_geom = bat_geom

            wall_width = 5
            for w in self.walls:
                x0, y0, x1, y1 = np.array(w) * scale
                l, r = x0 - wall_width/2, x1 + wall_width/2, 
                b, t = y0 - wall_width/2, y1 + wall_width/2
                wall_geom = rendering.FilledPolygon(
                    [(l, b), (l, t), (r, t), (r, b)])
                wall_geom.set_color(0.5, 0.5, 0.5)
                self.viewer.add_geom(wall_geom)
        
        bat_geom = self._bat_geom
        self.battrans.set_translation(
            self.bat.x * scale, self.bat.y * scale)
        self.battrans.set_rotation(self.bat.angle)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
