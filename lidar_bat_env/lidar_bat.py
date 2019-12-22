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
        self.body_weight = 23e-3 # [kg]
        self.size = 7e-2  # [m]
        self.n_memory = 5  # number of states
        self.state = np.zeros((self.n_memory, 2))

    def emit_pulse(self):
        pass

    def _cal_angle(self):
        # if self.v_x == 0:
        #     if self.v_y != 0:
        #         self.angle = math.pi/2 if self.v_y > 0 else -math.pi/2
        # else:
        #     self.angle = math.atan(self.v_y/self.v_x)
        self.angle = math.atan2(self.v_y, self.v_x)

    def move(self, acceleration, angle):
        a_x = acceleration * math.cos(self.angle + angle) 
        a_y = acceleration * math.sin(self.angle + angle)
        self.x += (self.v_x + 1/2 * a_x * self.dt) * self.dt
        self.y += (self.v_y + 1/2 * a_y * self.dt) * self.dt
        self.v_x += a_x * self.dt
        self.v_y += a_y * self.dt
        self._cal_angle()
    
    def bump(self, x0, y0, surface_angle, e=0.3):
        '''
        simulate partially inelastic collisions.
        e: coefficient of restitution
        '''
        sin, cos = math.sin(surface_angle), math.cos(surface_angle)
        v_to_surface = self.v_x * sin + self.v_y * cos
        v_along_surface = self.v_x * cos+ self.v_y * sin
        self.v_x = - e * v_to_surface * sin + v_along_surface * cos
        self.v_y = - e * v_to_surface * cos + v_along_surface * sin
        self.x = x0 + self.v_x * self.dt
        self.y = y0 + self.v_y * self.dt
        self._cal_angle()