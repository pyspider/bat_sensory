import math
import numpy as np


class LidarBat(object):
    def __init__(self, init_angle, init_x, init_y, init_speed, dt):
        self.angle = init_angle
        self.x = init_x  # [m]
        self.y = init_y  # [m]
        self.v_x = init_speed * math.cos(init_angle)  # [m/s]
        self.v_y = init_speed * math.sin(init_angle)  # [m/s]
        self.dt = dt  # [s]
        self.n_memory = 5  # number of states
        self.state = np.zeros((self.n_memory, 2))

    def emit_pulse(self):
        pass

    def move(self, acceleration, angle):
        self.angle += angle 
        a_x = acceleration * math.cos(self.angle)
        a_y = acceleration * math.sin(self.angle)
        self.x += (self.v_x + 1/2 * a_x * self.dt) * self.dt
        self.y += (self.v_y + 1/2 * a_y * self.dt) * self.dt
        self.v_x += a_x * self.dt
        self.v_y += a_y * self.dt