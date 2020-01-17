import math
import numpy as np


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def unpack(self):
        return np.array([self.x, self.y])

class Segment(object):
    def __init__(self, p0: Point, p1: Point):
        self.p0 = p0
        self.p1 = p1

    def unpack(self):
        return np.array([self.p0.x, self.p0.y, self.p1.x, self.p1.y])


def cos_sin(theta) -> np.ndarray:
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

def rotate_vector(v, angle):
    return np.array(
        [[np.cos(angle), -np.sin(angle)], 
         [np.sin(angle), np.cos(angle)]]) @ v

def is_point_in_segment(p: Point, s: Segment) -> bool:
    e = 1e-8  # e is small number, for excuse 
    x_ok = (min(s.p0.x, s.p1.x) - e <= p.x) and (p.x <= max(s.p0.x, s.p1.x) + e)
    y_ok = (min(s.p0.y, s.p1.y) - e <= p.y) and (p.y <= max(s.p0.y, s.p1.y) + e)
    return x_ok and y_ok

def convert2vec(v):
    if type(v) == Point:
        v = v.unpack()
    if type(v) == Segment:
        v = v.p1.unpack() - v.p0.unpack()
    return v

def cos_similarity(v0, v1):
    v0 = convert2vec(v0)
    v1 = convert2vec(v1)
    return np.inner(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))

def rotation_direction(v0, v1):
    v0 = convert2vec(v0)
    v1 = convert2vec(v1)
    outer = v0[0] * v1[1] - v1[0] * v0[1]
    return 1 if outer >= 0 else -1

class LidarBat(object):
    def __init__(self, init_angle, init_x, init_y, init_speed, dt):
        self.angle = init_angle
        self.x = init_x  # [m]
        self.y = init_y  # [m]
        self.bat_vec = np.array([self.x, self.y])
        self.v_x, self.v_y = init_speed * cos_sin(init_angle)  # [m/s]
        self.v_vec = np.array([self.v_x, self.v_y])
        self.dt = dt  # [s]

        self.body_weight = 23e-3 # [kg]
        self.size = 7e-2  # [m]

        self.n_memory = 5  # number of states
        self.state = np.array([[0, np.inf] for i in range(self.n_memory)])
        self.emit = False

        self.lidar_length = 10
        self.lidar_left_angle = (math.pi / 6) / 2
        self.lidar_right_angle = -(math.pi / 6) / 2
        self.lidar_range = np.array([
            self.lidar_left_angle, self.lidar_right_angle])  # [rad]
    

    def emit_pulse(self, lidar_angle, obstacle_segments):
        lidar_vec = cos_sin(lidar_angle) * self.lidar_length
        left_lidar_seg, right_lidar_seg = self._lidar_segments(lidar_vec)
        left_lidar_vec = left_lidar_seg.p1.unpack() - left_lidar_seg.p0.unpack()
        right_lidar_vec = right_lidar_seg.p1.unpack() - right_lidar_seg.p0.unpack()
        cs0 = cos_similarity(left_lidar_vec, right_lidar_vec)

        detected_points = []
        for s in obstacle_segments:
            edge_vec0 = s.p0.unpack() - self.bat_vec
            edge_vec1 = s.p1.unpack() - self.bat_vec

            cs1 = cos_similarity(edge_vec0, left_lidar_vec)
            cs2 = cos_similarity(edge_vec0, right_lidar_vec)
            cs3 = cos_similarity(edge_vec1, left_lidar_vec)
            cs4 = cos_similarity(edge_vec1, right_lidar_vec)

            left_c_p  = cal_cross_point(left_lidar_seg, s)
            right_c_p = cal_cross_point(right_lidar_seg, s)
            is_left_cross  = (is_point_in_segment(left_c_p, s) and 
                              is_point_in_segment(left_c_p, left_lidar_seg))
            is_right_cross = (is_point_in_segment(right_c_p, s) and 
                              is_point_in_segment(right_c_p, right_lidar_seg))

            new_seg = None
            if is_left_cross and is_right_cross:
                new_seg = Segment(left_c_p, right_c_p)
            elif is_left_cross:
                if cs1 > cs0 and cs2 > cs0:
                    new_seg = Segment(left_c_p, s.p0)
                if cs3 > cs0 and cs4 > cs0:
                    new_seg = Segment(left_c_p, s.p1)
            elif is_right_cross:
                if cs1 > cs0 and cs2 > cs0:
                    new_seg = Segment(s.p0, right_c_p)
                if cs3 > cs0 and cs4 > cs0:
                    new_seg = Segment(s.p1, right_c_p)
            else:
                if cs1 > cs0 and cs2 > cs0 and cs3 > cs0 and cs4 > cs0:
                    new_seg = s
            if new_seg is not None:
                detected_points.append(new_seg.p0)
                detected_points.append(new_seg.p1)
        
        min_length = np.inf
        nearest_point_vec = None
        for p in detected_points:
            detected_vec = p.unpack() - self.bat_vec
            detected_length = np.linalg.norm(detected_vec)
            if min_length > detected_length:
                min_length = detected_length
                nearest_point_vec = detected_vec
        if nearest_point_vec is None:
            observation = np.array([0, np.inf])
        else:
            observation = rotate_vector(nearest_point_vec, -self.angle) 
        observation /= self.lidar_length
        self._update_state(observation)
        return observation

    def _lidar_segments(self, lidar_vec):
        lidar_vec = rotate_vector(lidar_vec, self.angle)
        v_left    = rotate_vector(lidar_vec, self.lidar_left_angle)
        v_right   = rotate_vector(lidar_vec, self.lidar_right_angle)
        v_left, v_right = v_left + self.bat_vec, v_right + self.bat_vec
        bat_p = Point(*self.bat_vec)
        left_p = Point(*v_left)
        right_p = Point(*v_right)
        return Segment(bat_p, left_p), Segment(bat_p, right_p)

    def _update_state(self, new_observation):
        self.state[1:] = self.state[:-1]
        self.state[0] = new_observation

    def move(self, angle):
        self.v_vec = rotate_vector(self.v_vec, angle)
        self.bat_vec += self.v_vec * self.dt
        self._cal_angle()
    
    def bump(self, bat_vec, surface_vec, e=1):
        '''
        simulate partially inelastic collisions.
        e: coefficient of restitution
        '''
        T = surface_vec / np.linalg.norm(surface_vec) # Tangent vector
        N = np.array([-T[1], T[0]])  # Normal vector
        v_T = np.inner(self.v_vec, T) * T
        v_N = np.inner(self.v_vec, N) * N
        self.v_vec = -e * v_N + v_T
        self.bat_vec = bat_vec + self.v_vec * self.dt
        self._cal_angle()

    def _cal_angle(self):
        self.angle = math.atan2(*self.v_vec[::-1])