import math
import numpy as np


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def length_to_point(self, x, y):
        return np.linalg.norm([self.x - x, self.y - y])
    
    def unpack(self):
        return np.array([self.x, self.y])

class Segment(object):
    def __init__(self, p0: Point, p1: Point):
        self.p0 = p0
        self.p1 = p1

    def nearest_point(self, x, y, e=1e-8):
        length_to_p0 = self.p0.length_to_point(x, y)
        length_to_p1 = self.p0.length_to_point(x, y)
        if abs(length_to_p0 - length_to_p1) < e:
            x = (min(self.p0.x, self.p1.x) - max(self.p0.x, self.p1.x))/2
            y = (min(self.p0.y, self.p1.y) - max(self.p0.y, self.p1.y))/2
            return Point(x, y)
        if length_to_p0 < length_to_p1:
            return self.p0
        return self.p1

    def unpack(self):
        return np.array([self.p0.x, self.p0.y, self.p1.x, self.p1.y])
 

def cos_sin(theta):
    return np.array([math.cos(theta), math.sin(theta)])

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
    e = 1e-8  # e is small number, for excuse 
    x_ok = min(s.p0.x, s.p1.x) - e <= p.x and p.x <= max(s.p0.x, s.p1.x) + e
    y_ok = min(s.p0.y, s.p1.y) - e <= p.y and p.y <= max(s.p0.y, s.p1.y) + e
    return x_ok and y_ok


class LidarBat(object):
    def __init__(self, init_angle, init_x, init_y, init_speed, dt):
        self.angle = init_angle
        self.x = init_x  # [m]
        self.y = init_y  # [m]
        self.v_x, self.v_y = init_speed * cos_sin(init_angle)  # [m/s]
        self.dt = dt  # [s]

        self.body_weight = 23e-3 # [kg]
        self.size = 7e-2  # [m]

        self.n_memory = 5  # number of states
        self.state = np.zeros((self.n_memory, 2))

        self.lidar_length = 20
        self.lidar_left_angle = math.pi / 4
        self.lidar_right_angle = -math.pi / 4
        self.lidar_range = np.array([
            self.lidar_left_angle, self.lidar_right_angle])  # [rad]

    def emit_pulse(self, angle, obstacle_segments):
        for clipped_segment in map(self._clip_segment_lidar_range, obstacle_segments):
            if clipped_segment is None:
                continue
            clipped_segment.p0.x
        # pass

    def _angle_from_bat(self, x, y):
        return math.atan2(y - self.y, x - self.x) - self.angle

    def _clip_segment_lidar_range(self, s: Segment) -> Segment:
        a0 = self._angle_from_bat(s.p0.y, s.p0.x) 
        a1 = self._angle_from_bat(s.p1.y, s.p1.x)
        if a0 > a1:
            left_angle, right_angle = a0, a1
            left_point, right_point = s.p0, s.p1
        else:
            left_angle, right_angle = a1, a0
            left_point, right_point = s.p1, s.p0
        left_angle = max(a0, a1)
        right_angle = min(a0, a1)
        if (   left_angle  <= self.lidar_right_angle
            or right_angle >= self.lidar_left_angle
            or left_angle - right_anlge >= math.pi / 2):
            return None
        p_nose = Point(self.x, self.y)
        if left_angle > self.lidar_left_angle:
            left_point = self._cross_point_lidar_and_segment(left_angle, s)
        if right_angle < self.lidar_right_angle:
            right_point = self._cross_point_lidar_and_segment(right_angle, s)
        return Segment(left_point, right_point)
    
    def _cross_point_lidar_and_segment(self, lidar_angle, segment) -> Point:
        x, y = self.lidar_length * cos_sin(lidar_angle)
        p_nose = Point(self.x, self.y)
        p_lidar = Point(self.x + x, self.y + y)
        lidar_seg = Segment(p_nose, p_lidar)
        c_p = cal_cross_point(s, lidar_seg)
        if is_point_in_segment(c_p, s) is True:
            return c_p
        else:
            print('warnings: cross point is not in segment.')

    def _cal_angle(self):
        self.angle = math.atan2(self.v_y, self.v_x)

    def move(self, acceleration, angle):
        # a_x = acceleration * math.cos(self.angle + angle) 
        # a_y = acceleration * math.sin(self.angle + angle)
        a_x, a_y = acceleration * cos_sin(self.angle + angle)
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
        # sin, cos = math.sin(surface_angle), math.cos(surface_angle)
        cos, sin = cos_sin(surface_angle)
        v_to_surface = self.v_x * sin + self.v_y * cos
        v_along_surface = self.v_x * cos+ self.v_y * sin
        self.v_x = - e * v_to_surface * sin + v_along_surface * cos
        self.v_y = - e * v_to_surface * cos + v_along_surface * sin
        self.x = x0 + self.v_x * self.dt
        self.y = y0 + self.v_y * self.dt
        self._cal_angle()