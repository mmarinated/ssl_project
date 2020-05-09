import numpy as np
import torch
from ssl_project.constants import CAM_NAMES, CAM_ANGLE
import cv2
import kornia
from ssl_project.preprocessing import projections

n_W = 800

class ProjectionToBEV:
    def __init__(self, camera_height_in_m=1.7, alpha=70, level_height_in_m=0):
        self.CAM_matrices_6_3_3, self.h_size = self._get_projection_matrices_and_h(
            camera_height_in_m, alpha, level_height_in_m,
        )
        
    
    def _get_projection_matrices_and_h(
            self, camera_height_in_m, alpha, level_height_in_m,
        ):
        """
        returns CAM_matrices_6_3_3, h_size
        """
        corners = projections.Corners(camera_height_in_m=camera_height_in_m, 
                                      alpha=alpha, level_height_in_m=level_height_in_m)

        photo_corners_yz_42, map_corners_xy_42 = corners.get_photo_and_map_corners()
        points_src, points_tgt_42 = photo_corners_yz_42.astype("float32"), map_corners_xy_42.astype("float32")
        h_size = int(points_src[0, 1] - points_src[-1, 1])

        CAM_matrices_6_3_3 = []
        for angle in CAM_ANGLE:
            points_tgt_rotated_42 = self._Rotate2D(points_tgt_42, (n_W // 2, n_W // 2), ang=-angle).astype("f4")
#             points_tgt_rotated_42[:, 0] -= 30
            M = cv2.getPerspectiveTransform(points_src, points_tgt_rotated_42)
            CAM_matrices_6_3_3.append(M)
        return CAM_matrices_6_3_3, h_size
    
    def _clip_photo(self, photo_3hw, return_up, offset):
        test_photo_3hw = photo_3hw.clone()
        if return_up:
            test_photo_3hw[..., -(test_photo_3hw.shape[1] - self.h_size - offset):, :] = 0
    #         test_photo_3hw = torch.flip(test_photo_3hw, (1, 2))
        else:
            test_photo_3hw[..., :(test_photo_3hw.shape[1] - self.h_size - offset), :] = 0    
        return test_photo_3hw

    @staticmethod
    def _Rotate2D(pts, cnt, ang):
        ang = np.radians(ang)
        rotation_matrix = np.array([
            [ np.cos(ang), np.sin(ang)],
            [-np.sin(ang), np.cos(ang)]
        ])

        return np.dot(pts-cnt, rotation_matrix) + cnt

    def get_warped_3WW(self, photos_63hw, return_up=False, offset=5):
        img_warp_63WW = torch.zeros(6, 3, n_W, n_W)
        for idx, (M, photo_3hw) in enumerate(zip(self.CAM_matrices_6_3_3, photos_63hw)):
            img_warp_63WW[idx] = kornia.warp_perspective(
                self._clip_photo(photo_3hw[None], return_up=return_up, offset=offset),
                torch.Tensor(M), dsize=(n_W, n_W))[0]


        count_3WW = (img_warp_63WW > 0).sum(0)
        mean_img_warp_3WW = img_warp_63WW.sum(0) / (1e-6 + count_3WW)
        return torch.flip(mean_img_warp_3WW, (1, 2)) if return_up else mean_img_warp_3WW