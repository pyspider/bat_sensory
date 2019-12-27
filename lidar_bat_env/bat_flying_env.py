from collections import namedtuple
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from .lidar_bat import *


FPS = 60


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

        margin = 0.1
        p0 = Point(margin, margin)
        p1 = Point(margin, world_height - margin)
        p2 = Point(world_width - margin, world_height - margin)
        p3 = Point(world_width - margin, margin)
        w0 = Segment(p0, p1)
        w1 = Segment(p1, p2)
        w2 = Segment(p2, p3)
        w3 = Segment(p3, p0)
        walls = [w0, w1, w2, w3]
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
        
        self.default_bat = lambda: LidarBat(0, 0.3, 0.75, 3, self.dt)
        self.bat = self.default_bat() if bat is None else bat
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
            c_p = cal_cross_point(bat_seg, w)
            if is_point_in_segment(c_p, bat_seg) == True:
                wall_angle = math.atan2(w.p1.y - w.p0.y, w.p1.x - w.p0.x)
                self.bat.bump(bat_p0.x, bat_p0.y, wall_angle)
                step_reward = -1.0

        if action[2] >= 0.5:
            self.bat.emit_pulse(action[3], self.walls)

        done = None
        step_reward = 0
        return np.array(self.bat.state), step_reward, done, {}

    def reset(self, bat=None):
        self.bat = self.default_bat() if bat is None else bat
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
                x0, y0, x1, y1 = w.unpack() * scale
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
