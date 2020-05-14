
# import torch
# import torchvision
# import numpy as np

# from tqdm import tqdm

# from constants import LABELED_SCENE_INDEX
# from data_loaders.data_helper import LabeledDataset
# from paths import PATH_TO_ANNOTATION, PATH_TO_DATA
# from preprocessing import projections
# from constants import CAM_NAMES


# def main():
#     transform = torchvision.transforms.ToTensor()

#     # The labeled dataset can only be retrieved by sample.
#     # And all the returned data are tuple of tensors, since bounding boxes may have different size
#     # You can choose whether the loader returns the extra_info. It is optional. You don't have to use it.
#     labeled_trainset = LabeledDataset(image_folder=PATH_TO_DATA,
#                                     annotation_file=PATH_TO_ANNOTATION,
#                                     scene_index=LABELED_SCENE_INDEX,
#                                     transform=transform,
#                                     extra_info=True
#                                     )

#     annotate = projections.PixelLabels(x_shift=0)

#     for i in tqdm(range(len(labeled_trainset))):
#         sample, target, road_image, extra = labeled_trainset[i]
#         _, _, sample_path = labeled_trainset.get_ids_and_path(i)

#         mask_6hw = annotate.get_pixel_segmentation_for_photos(target)
#         for cam_name, mask_hw in zip(CAM_NAMES, mask_6hw):
#             np.save(f"{sample_path}/{cam_name}", mask_hw)

# if __name__ == "__main__":
#     main()