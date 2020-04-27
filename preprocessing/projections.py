import matplotlib.cm as cm
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import torchvision
from ssl_project.utils import to_np
from ssl_project.data_loaders.helper import convert_map_to_road_map

import cv2
from ssl_project.constants import *
from ssl_project.paths import *
from ssl_project.preprocessing import top_down_segmentation

VALUE_TO_CATEGORY = top_down_segmentation.VALUE_TO_CATEGORY
GROUND_CAT = 13
assert VALUE_TO_CATEGORY[GROUND_CAT] == 'ground'


NO_IMAGE_LABEL = -1

__all__ = ['create_label_data', 'create_data_all']


class projection_func:  
    """used to get corners etc, for images we use cv2"""
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



def rotate_image(image, angle, borderValue):
    """rotates with padding"""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result  = cv2.warpAffine(image, rot_mat, image.shape[1::-1], 
                             flags=cv2.INTER_NEAREST, borderValue=borderValue)
    return result


class Corners:
    def __init__(self, camera_height_in_m, alpha, level_height_in_m=0, *,
                 image_dims=(PHOTO_W, PHOTO_H)):
        """
        Assumes dist to projection plane is 1.
        
        height = horizontal plane (e.g. road) height
        """
        assert level_height_in_m < camera_height_in_m
        
        self.camera_height  = camera_height_in_m * 10
        self.angle_view     = np.radians(alpha)
        self.level_height   = level_height_in_m * 10
        
        self.image_width, self.image_height = image_dims
        
        self.view_point = np.array([0, 0, self.camera_height])
        
        self._f_xyz_to_camera_plane = projection_func(
            self.view_point,
            np.array([1, 0, 0,]),
            np.array([1, 0, 0,]),
        )

        self._f_camera_to_xy_h = projection_func(
            self.view_point,
            np.array([0, 0, self.level_height]),
            np.array([0, 0, 1,]),
        )
    

    def _scale(self, yz_42):
        new_yz_42 = yz_42.copy()
        new_yz_42[..., 0] = self._y_axis_transform.predict(yz_42[..., 0:1])
        new_yz_42[..., 1] = self._z_axis_transform.predict(yz_42[..., 1:2])

        return new_yz_42


    def get_photo_and_map_corners(self):
        """Returns photo_corners_43, map_corners_43"""
        photo_down_corners_23 = self._get_photo_down_corners()
        map_back_corners_23  = self._get_map_back_corners()
        photo_up_corners_23  = self._get_photo_up_corners(map_back_corners_23)
        map_front_corners_23 = self._get_map_front_corners(photo_down_corners_23)
        
        tmp_photo_corners_43, tmp_map_corners_43 = (
            np.vstack([photo_down_corners_23, photo_up_corners_23]), 
            np.vstack([map_front_corners_23, map_back_corners_23]),
        )
        
        photo_corners_yz_42 = self._scale(tmp_photo_corners_43[:, 1:])
        map_corners_xy_42 = tmp_map_corners_43[:, :2] + 400
        return photo_corners_yz_42.astype(int), map_corners_xy_42.astype(int)

    def _get_photo_down_corners(self, dist_to_plane=1):
        y_diff = np.tan(self.angle_view / 2) * dist_to_plane
        z_diff = y_diff * self.image_height / self.image_width
    
        def get_linear_tansform(x1, x2, new_x1, new_x2):
            lr = LinearRegression()
            lr.fit([[x1], [x2]], [new_x1, new_x2])
            return lr

    
        self._y_axis_transform = get_linear_tansform(
            - y_diff, y_diff, 0, self.image_width)
        self._z_axis_transform = get_linear_tansform(
            self.camera_height + z_diff, 
            self.camera_height - z_diff, 
            0, self.image_height)
    
        return np.array([
            [1, - y_diff, self.camera_height - z_diff],
            [1, + y_diff, self.camera_height - z_diff],
        ])

    def _get_map_back_corners(self):
        dist_far = 400 # dist from center to border
        y_diff = np.tan(self.angle_view / 2) * dist_far
        return np.array([
            [dist_far, - y_diff, self.level_height],
            [dist_far, + y_diff, self.level_height],
        ])

    def _get_photo_up_corners(self, map_back_corners_23):
        return self._f_xyz_to_camera_plane(map_back_corners_23)

    def _get_map_front_corners(self, photo_down_corners_23):
        return self._f_camera_to_xy_h(photo_down_corners_23)


def get_warped_hw(
        segm_WW, height_WW, *, 
        camera_idx, level_height_in_m,
        camera_height_in_m = 1.7):
    
    segm_WW    = rotate_image(segm_WW, -IDX_TO_ANGLE[camera_idx],   borderValue=GROUND_CAT)
    height_WW  = rotate_image(height_WW, -IDX_TO_ANGLE[camera_idx], borderValue=0)

    

    if level_height_in_m > camera_height_in_m:
        level_height_in_m_fake = 2 * camera_height_in_m - level_height_in_m 
        inverse = True
    else:
        level_height_in_m_fake = level_height_in_m
        inverse = False
    
    corners = Corners(camera_height_in_m=camera_height_in_m, 
                      alpha=VIEW_ANGLE, level_height_in_m=level_height_in_m_fake)
    photo_corners_yz_42, map_corners_xy_42 = corners.get_photo_and_map_corners()

    
    points_src, points_tgt = map_corners_xy_42.astype("float32"), photo_corners_yz_42.astype("float32")
    h_size = int(points_tgt[0, 1] - points_tgt[-1, 1])

    M = cv2.getPerspectiveTransform(points_src, points_tgt)
    
    def get_warped_image(image_WW):
        return cv2.warpPerspective(image_WW[:, ::-1], M, (PHOTO_W, h_size), flags=cv2.INTER_NEAREST).astype(int)
    
    warped_segm    = get_warped_image(segm_WW)
    warped_height  = get_warped_image(height_WW)
    
    warped_segm[warped_height < level_height_in_m] = NO_IMAGE_LABEL
    answer_hw = np.full((PHOTO_H, PHOTO_W), NO_IMAGE_LABEL)
    answer_hw[PHOTO_H - h_size:] = warped_segm[::-1, ::-1]
    
    return answer_hw[::-1] if inverse else answer_hw



def create_height_WW(segm_WW):
    height_WW = np.zeros_like(segm_WW)
    for val, category in top_down_segmentation.VALUE_TO_CATEGORY.items():
        if category in CATEGORY_TO_HEIGHT:
            height = CATEGORY_TO_HEIGHT[category]
            height_WW[segm_WW == val] = height

    return height_WW


def create_dist_WW():
    dist_WW = np.zeros((800, 800))
    dist_WW += ((np.arange(800) - 400)**2)[None]
    dist_WW += ((np.arange(800) - 400)**2)[:, None]
    dist_WW /= 100
    dist_WW = np.sqrt(dist_WW)
    return dist_WW


def create_segm_hw(top_down_segm_WW, height_WW, camera_idx):
    """returns segm_hw with NO_IMAGE_LABEL """
    segm_hw     = get_warped_hw(top_down_segm_WW, height_WW, 
                                camera_idx=camera_idx, level_height_in_m=0)

    for height in np.arange(0, height_WW.max(), 0.05): # TODO -- main speed up should be here
        tmp_hw     = get_warped_hw(top_down_segm_WW, height_WW,  
                                    camera_idx=camera_idx, level_height_in_m=height)
        np.copyto(segm_hw, tmp_hw, where=(tmp_hw != NO_IMAGE_LABEL))

    # HACK HERE
    # idea is that we have empty spots near 1.7 let's fill it dirty way 
    for height in np.arange(1.6, min(height_WW.max(), 1.8), 0.025): # TODO -- main speed up should be here
        tmp_hw     = get_warped_hw(top_down_segm_WW, height_WW,  
                                    camera_idx=camera_idx, level_height_in_m=height,
                                    camera_height_in_m=2)
        np.copyto(segm_hw, tmp_hw, where=(tmp_hw != NO_IMAGE_LABEL))


    return segm_hw  


def create_label_data(scene_id):
    dist_WW = create_dist_WW()  

    
    for sample_id in tqdm(range(NUM_SAMPLE_PER_SCENE)):
        PATH_TMP = f"{PATH_TO_DATA}/scene_{scene_id}/sample_{sample_id}"
        top_down_segm_WW = cv2.imread(f"{PATH_TMP}/top_down_segm.png", cv2.IMREAD_UNCHANGED)
        height_WW = create_height_WW(top_down_segm_WW)
        
        for idx, camera_name in enumerate(CAM_NAMES):
            # photo_hw = cv2.imread(f"../data/scene_{scene_id}/sample_{sample_id}/{name}.jpeg", cv2.IMREAD_UNCHANGED)
            segm_hw = create_segm_hw(top_down_segm_WW, height_WW, idx)
            segm_hw[segm_hw == NO_IMAGE_LABEL] = max(top_down_segmentation.VALUE_TO_CATEGORY.keys()) + 1
            dist_hw = create_segm_hw(dist_WW,          height_WW, idx)

            cv2.imwrite(f"{PATH_TMP}/SEGM_{camera_name}.png", segm_hw)
            cv2.imwrite(f"{PATH_TMP}/DIST_{camera_name}.png", dist_hw)



def create_label_data_road_only(scene_id):
    def get_road_image(PATH_TMP):
        ego_image = cv2.imread(f"{PATH_TMP}/ego.png", cv2.IMREAD_UNCHANGED)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = to_np(convert_map_to_road_map(ego_image)).astype(int)
        return road_image

    height_WW = np.zeros((EGO_IMAGE_SIZE, EGO_IMAGE_SIZE))
    for sample_id in tqdm(range(NUM_SAMPLE_PER_SCENE)):
        PATH_TMP = f"{PATH_TO_DATA}/scene_{scene_id}/sample_{sample_id}"
        top_down_segm_WW = get_road_image(PATH_TMP) # TODO cv2.imread(f"{PATH_TMP}/top_down_segm.png", cv2.IMREAD_UNCHANGED)
        # return top_down_segm_WW
        
        for idx, camera_name in enumerate(CAM_NAMES):
            segm_hw = create_segm_hw(top_down_segm_WW, height_WW, idx)
            # 1 is road 0 is not road
            segm_hw[segm_hw != 1] = 0 # TODO max(top_down_segmentation.VALUE_TO_CATEGORY.keys()) + 1
            cv2.imwrite(f"{PATH_TMP}/ROAD_SEGM_{camera_name}.png", segm_hw)
            


def create_data_all(n_jobs=8, slc=slice(None), debug=False):    
    if debug:
        for idx in tqdm(LABELED_SCENE_INDEX[slc]):
            create_label_data(idx)
    else:
        Parallel(n_jobs=n_jobs)(
            delayed(create_label_data)(scene_id) 
            for scene_id in LABELED_SCENE_INDEX[slc]
        )
