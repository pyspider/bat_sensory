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
        2     Emit Pulse
        3     Pulse direction

    Reward:
        Reword is 1 for every step take, including the termination step

    Starting State:
        position
        direction
        speed

    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    def __init__(
            self,
            world_width=1.5,
            world_height=1.5,
            discrete_length=0.01,
            dt=0.005,
            bat=None,
            walls=None,
            goal_area=None,
            max_accel=None,
            max_accel_angle=None,
            max_pulse_angle=None):
        # world settings
        self.world_width = world_width
        self.world_height = world_height
        self.discrete_length = discrete_length
        self.dt = 0.005  # [s]

        self.accel_reward = 0
        self.accel_angle_reward = 0
        self.pulse_reward = 0
        self.pulse_angle_reward = 0
        self.bump_reward = -100.0
        self.fliyng_reward = 1

        # walls settings
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

        # bat settings
        self.default_bat = lambda: LidarBat(0, 0.3, 0.75, 3, self.dt)
        self.bat = self.default_bat() if bat is None else bat

        # self.goal_area = () if goal_area is None else goal_area
        self.max_accel = 50  # [m/s^2]
        self.max_accel_angle = math.pi / 2 # [rad]
        self.max_pulse_angle = math.pi / 4 # [rad]

        # env settings
        self.action_space = spaces.Box(
            np.array([
                -1.0,
                -1.0,
                0,
                -1.0]),
            np.array([
                1.0,
                1.0,
                1.0,
                1.0]),
            dtype=np.float32)
        
        self.max_echo_distance = 10
        self.observation_space = spaces.Box(
            np.zeros(2),
            np.array([self.max_echo_distance, 1]),
            dtype=np.float32)
        
        self.viewer = None
        self.state = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
   
    def step(self, action):
        step_reward = 0
        done = False
        accel, accel_angle, pulse_proba, pulse_angle = action
        step_reward += self.accel_reward * accel
        step_reward += self.accel_angle_reward * np.abs(accel_angle)

        bat_p0 = Point(self.bat.x, self.bat.y)
        self.bat.move(accel * self.max_accel, accel_angle * self.max_accel_angle)
        bat_p1 = Point(self.bat.x, self.bat.y)
        bat_seg = Segment(bat_p0, bat_p1)
        for w in self.walls:
            c_p = cal_cross_point(bat_seg, w)
            if is_point_in_segment(c_p, bat_seg) == True:
                wall_angle = math.atan2(w.p1.y - w.p0.y, w.p1.x - w.p0.x)
                self.bat.bump(bat_p0.x, bat_p0.y, wall_angle)
                step_reward += self.bump_reward
                done = True

        self.bat.emit = False
        if np.random.rand() > pulse_proba:
            self.bat.emit_pulse(pulse_angle * self.max_pulse_angle, self.walls)
            self.bat.emit = True
            self.last_pulse_angle = pulse_angle * self.max_pulse_angle
            step_reward += self.pulse_angle_reward * np.abs(pulse_angle) 
            step_reward += self.pulse_reward

        self.t += self.dt
        self.state = np.array(self.bat.state)
        return self.state, step_reward, done, {}

    def reset(self):
        self.bat = self.default_bat()
        self.t = 0.0
        self.state = np.array(self.bat.state)
        self.close()
        return self.state

    def render(self, mode='human', screen_width=500):
        # whether draw pulse and echo source
        draw_pulse_direction = True
        draw_echo_source = True

        # settings screen
        aspect_ratio = self.world_height / self.world_width
        screen_height = int(aspect_ratio * screen_width)
        scale = screen_width / self.world_width

        # initilize screen
        from gym.envs.classic_control import rendering
        if self.viewer is None:
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

            wall_width = 5  # pixel
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

        if self.bat.emit == True: 
            if draw_pulse_direction == True:
                pulse_length = 0.5
                bat_vec = np.array([self.bat.x, self.bat.y])
                pulse_vec = pulse_length * cos_sin(self.last_pulse_angle)
                pulse_vec = rotate_vector(pulse_vec, self.bat.angle) + bat_vec
                x0, y0 = bat_vec * scale
                x1, y1 = pulse_vec * scale
                line = self.viewer.draw_line([x0, y0], [x1, y1])
                self.viewer.add_geom(line)

            if draw_echo_source == True:
                radius = 4  # pixel
                l, a = self.bat.state[0]
                echo_source_vec = l * cos_sin(a)
                echo_source_vec = rotate_vector(echo_source_vec, self.bat.angle) + bat_vec
                x, y = echo_source_vec * scale
                echo_source = rendering.make_circle(radius)
                echo_source.set_color(0.9, 0.65, 0.4)
                echotrans = rendering.Transform()
                echo_source.add_attr(echotrans)
                echotrans.set_translation(x, y)
                self.viewer.add_geom(echo_source)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
