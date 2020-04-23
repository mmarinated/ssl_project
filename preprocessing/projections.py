"""
utility functions to project from 2d plane to another
"""

import numpy as np
from ssl_project.utils import to_np
from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay
from ssl_project.preprocessing import projections
from paths import PATH_TO_REPO

from constants import IDX_TO_ANGLE, PHOTO_H, PHOTO_W, VIEW_ANGLE


class PixelLabels:
    def __init__(self, camera_height=1.7, alpha=VIEW_ANGLE, x_shift=0, *, 
                 image_dims=(PHOTO_W, PHOTO_H)):
        self.camera_height = camera_height
        self.alpha = alpha
        self.x_shift = x_shift
        self.n_w, self.n_h = image_dims
        x, y = np.meshgrid(np.arange(self.n_w), np.arange(self.n_h))
        self.xy_N2 = np.vstack((x.ravel(), y.ravel())).T

    @staticmethod
    def _get_cars_n(target):
        cars_n = np.array([
            to_np(target["bounding_box"][car_idx].T)
            for car_idx in range(len(target["bounding_box"]))
        ])
        
        return cars_n

    def get_pixel_segmentation_for_photos(self, target):
        """
        - assumes all cars are of height 1.5 m
        - no roads
        - no types

        Returns
        -------
        mask_6hw
        """
        cars_n = self._get_cars_n(target)

        target_6hw = np.zeros((6, self.n_h, self.n_w))

        for photo_idx in range(6):
            proj_plane = projections.ProjectionPlane(
                self.camera_height, self.alpha, self.x_shift, 
                IDX_TO_ANGLE[photo_idx])
            valid_n = proj_plane.is_valid(cars_n)

            for yz_42, yz_top_42 in zip(proj_plane(cars_n[valid_n]), 
                                        proj_plane(cars_n[valid_n], 1.5)):
                points = np.vstack((yz_42, yz_top_42))
                hull = Delaunay(points)
                mask_hw = (hull.find_simplex(self.xy_N2) >= 0).reshape(self.n_h, self.n_w)
                target_6hw[photo_idx][mask_hw] = 1
        
        return target_6hw


class ProjectionPlane:
    """
    # image # 57 in labeled_dataset
    >>> proj_plane = ProjectionPlane(1.7, 70, IDX_TO_ANGLE[1])
    >>> cars_n  = np.load(f"{PATH_TO_REPO}/tests/cars.npy")
    >>> valid_n = proj_plane.is_valid(cars_n)
    >>> yz_k42  = proj_plane(cars_n[valid_n])
    >>> yz_k42[0]
    array([[ 54.38444167, 139.41474844],
           [ 48.16907255, 140.11217843],
           [ 25.32221255, 139.39299565],
           [ 17.34732646, 140.08768914]])
    >>> valid_n.size, valid_n.sum()
    (21, 5)
    """
    
    def __init__(self, camera_height, alpha, x_shift=0, rotation_angle=0, *, 
                 image_dims=(PHOTO_W, PHOTO_H)):
        self.camera_height  = camera_height
        self.angle_view     = np.radians(alpha)
        self.x_shift        = x_shift
        self.rotation_angle = np.radians(rotation_angle)
        self.image_width, self.image_height = image_dims
    
        self._init_projection_plane()
        self._init_borders()
        self._init_scaler()

    def _init_projection_plane(self):
        self._view_point = (self.x_shift, 0, self.camera_height) # maybe TODO 
    
        # TODO based on rotation alpha
        x, y = np.cos(self.rotation_angle), np.sin(self.rotation_angle)
        self._norm_vector_to_plane      = (x, y, 0) # should not add x_shift
        self._point_in_projection_plane = (x + self.x_shift, y, self.camera_height)
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
        func = projection_func(
            np.array(self._view_point),
            np.array(self._point_in_projection_plane), 
            np.array(self._norm_vector_to_plane))

        for idx, (x_topdown, y_topdown) in enumerate(xy_42):
            rot_x, rot_y, yz_42[idx][1] = func(np.array([x_topdown, y_topdown, height])[None])[0]
            
            rot_x -= self.x_shift
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


        # at_least_one_visible = False
        # for (y, z) in yz_42:
        #      at_least_one_visible |= self._is_visible(y, z)   

        # is_valid &= at_least_one_visible

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


class projection_func:
    def __init__(self, view_3, in_plane_3, orth_3):
        self.view_3 = view_3
        self.in_plane_3 = in_plane_3
        self.orth_3 = orth_3
        self.coef = -np.dot(orth_3, view_3 -  in_plane_3)

    def __call__(self, p_n3):
        diff_n3 = p_n3 - self.view_3
        dot_n   = diff_n3.dot(self.orth_3)
        assert dot_n.ndim == 1
        diff_n3  *= (self.coef / dot_n)[:, None]

        return self.view_3 + diff_n3
