
import numpy as np
from sklearn.linear_model import LinearRegression


class ProjectionPlane:
    def __init__(self, camera_height, alpha, rotation_angle=0, image_dims=(306, 256)):
        self.camera_height  = camera_height
        self.angle_view     = np.radians(alpha)
        self.rotation_angle = np.radians(rotation_angle)
        self.image_width, self.image_height = image_dims
    
        self._init_projection_plane()
        self._init_borders()
        self._init_scaler()

    def _init_projection_plane(self):
        self._view_point = (0, 0, self.camera_height) # maybe TODO 
    
        # TODO based on rotation alpha
        x, y = np.cos(self.rotation_angle), np.sin(self.rotation_angle)
        self._norm_vector_to_plane      = (x, y, 0)
        self._point_in_projection_plane = (x, y, self.camera_height)
        assert (0.9999 < x**2 + y**2 < 1.0001) # dist to projection plane

    def _init_borders(self):
        """
        Assumes (0, 0) is the center
        """
        # TODO
        
        self._border_w = np.tan(self.angle_view / 2) # multiply by dist to projection plane, which is 1
        self._border_h = self.image_height / self.image_width * self._border_w

        # TODO REMOVE
        self.borders_42 = np.array([
            [self._border_w,  self.camera_height + self._border_h],
            [self._border_w,  self.camera_height - self._border_h],
            [-self._border_w, self.camera_height + self._border_h],
            [-self._border_w, self.camera_height - self._border_h]
        ])


    def _init_scaler(self):
        def get_linear_tansform(x1, x2, new_x1, new_x2):
            lr = LinearRegression()
            lr.fit([[x1], [x2]], [new_x1, new_x2])
            return lr

        self._x_axis_transform = get_linear_tansform(
            -self._border_w, self._border_w,
            0, self.image_width)
        self._y_axis_transform = get_linear_tansform(
            self.camera_height - self._border_h, 
            self.camera_height + self._border_h, 
            0, self.image_height)


    def __call__(self, xy_n42, height=0.):
        """
        Return
        ------
        yz_n42
        """
        yz_n42 = np.zeros_like(xy_n42)
        for n_idx, xy_42 in enumerate(xy_n42):
            yz_n42[n_idx] = self._get_bounding_box_helper(xy_42, height)
        return yz_n42


    def _get_bounding_box_helper(self, xy_42, height):
        yz_42 = np.zeros((4, 2)) # (y, z)
        for idx, (x_topdown, y_topdown) in enumerate(xy_42):
            rot_x, rot_y, yz_42[idx][1] = _isect_line_plane_v3(
                self._view_point, (x_topdown, y_topdown, height), 
                self._point_in_projection_plane, self._norm_vector_to_plane)
            
            x, yz_42[idx][0] = _rotate((rot_x, rot_y), -self.rotation_angle)
            assert 0.9999 < x < 1.0001

        yz_42 = self._scale(yz_42)
        return yz_42

    def _scale(self, yz_42):
        new_yz_42 = yz_42.copy()
        # NOTE: flip width and height to match the photo
        new_yz_42[..., 0] = self.image_width - self._x_axis_transform.predict(yz_42[..., 0:1])
        new_yz_42[..., 1] = self.image_height - self._y_axis_transform.predict(yz_42[..., 1:2])

        return new_yz_42


    def is_valid(self, xy_n42, height=0):
        """checks if ALL corners are visible and front"""
        return np.array([self._is_valid_helper(xy_42, height) for xy_42 in xy_n42])

    def _is_valid_helper(self, xy_42, height):
        is_valid = self._is_front(*xy_42[0])
        yz_42 = self(xy_42[None], height)[0]

        for (y, z) in yz_42:
             is_valid &= self._is_visible(y, z)   

        return is_valid

    def _is_visible(self, y, z):
        return (0 < y < self.image_width) and (0 < z < self.image_height)

    def _is_front(self, x, y):
        dist_1  = x**2 + y**2
        dist_2 = ((self._point_in_projection_plane[0] - x)**2
                + (self._point_in_projection_plane[1] - y)**2)

        return dist_1 > dist_2


import math

def _rotate(point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = (0, 0)
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def _isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """
    # generic math functions
    # ----------------------

    def add_v3v3(v0, v1):
        return (
            v0[0] + v1[0],
            v0[1] + v1[1],
            v0[2] + v1[2],
            )


    def sub_v3v3(v0, v1):
        return (
            v0[0] - v1[0],
            v0[1] - v1[1],
            v0[2] - v1[2],
            )


    def dot_v3v3(v0, v1):
        return (
            (v0[0] * v1[0]) +
            (v0[1] * v1[1]) +
            (v0[2] * v1[2])
            )


    def len_squared_v3(v0):
        return dot_v3v3(v0, v0)


    def mul_v3_fl(v0, f):
        return (
            v0[0] * f,
            v0[1] * f,
            v0[2] * f,

        )
    # ----------------------

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        return add_v3v3(p0, u)
    else:
        # The segment is parallel to plane.
        return None

