import matplotlib.pyplot as plt
import torchvision
import numpy as np
from ssl_project.data_loaders.helper import draw_box

from ssl_project.utils import to_np

def plot_photos(photos_n, axis=None, **kwargs):
    """
    Plots 1-6 photos.

    Input:
        tensor of size torch.Size([n, 3, 256, 306])
    
    Note:
        The 6 images orgenized in the following order:
        CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT
    """
    if axis is None:
        fig, axis = plt.subplots()
    axis.imshow(
        torchvision.utils.make_grid(photos_n, nrow=3).numpy().transpose(1, 2, 0),
        **kwargs
    )
    # plt.axis('off')

def plot_road(road_image, axis=None):
    """
    Plots road map.

    Input:
        tensor of size torch.Size([800, 800])

    Note:
        The road map layout is encoded into a binary array of size [800, 800] per sample 
        Each pixel is 0.1 meter in physiscal space, so 800 * 800 is 80m * 80m centered at the ego car
        The ego car is located in the center of the map (400, 400) and it is always facing the left
    """
    if axis is None:
        fig, axis = plt.subplots()
    axis.imshow(road_image, cmap='binary')
    return axis

def plot_bb(road_image, target_b, b_slc=slice(None), color=None, axis=None):
    """
    color
        if None plots category colors. Pass your own to override
    
    Note:
        The center of image is 400 * 400
    """
    if axis is None:
        fig, axis = plt.subplots()
    color_list = ['b', 'g', 'orange', 'c', 'm', 'y', 'k', 'w', 'r']
    if road_image is not None:
        plot_road(road_image, axis)
    # The ego car position
    axis.plot(400, 400, 'x', color="red")

    if isinstance(target_b, dict):
        bb_and_category = list(zip(to_np(target_b['bounding_box']), to_np(target_b['category'])))
    else:
        bb_and_category = list(zip(to_np(target_b), np.ones(len(target_b), dtype=int)))
        
    
    for bb, typ in np.array(bb_and_category)[b_slc]:
        draw_box(axis, bb, color=(color_list[typ] if color is None else color))


def draw_box(ax, corners, color):
    """description"""
    try:
        point_squence = torch.stack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])
    except:
        point_squence = np.vstack([corners[:, 0], corners[:, 1], corners[:, 3], corners[:, 2], corners[:, 0]])

    # the corners are in meter and time 10 will convert them in pixels
    # Add 400, since the center of the image is at pixel (400, 400)
    # The negative sign is because the y axis is reversed for matplotlib
    ax.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color=color)




