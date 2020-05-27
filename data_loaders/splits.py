"""implements scene (group) splitting"""

from ssl_project.constants import *
from ssl_project.paths import *
from ssl_project.utils import TRANSFORM

from ssl_project.data_loaders.data_helper import LabeledDataset



def get_train_val_test_ds(
    val_mask, test_mask=None, *,
    dataset_cls=LabeledDataset,
    image_folder=PATH_TO_DATA, annotation_file=PATH_TO_ANNOTATION,
    transform=None, extra_info=False, **kwargs,
):
    """
    return 
        train_labeled_set, val_labeled_set, test_labeled_set
    """
    assert test_mask is None, "it is always None => test_ds == None, change code if you want"

    val_idces = LABELED_SCENE_INDEX[val_mask]
    train_idces = LABELED_SCENE_INDEX[~np.isin(LABELED_SCENE_INDEX, val_idces)]
    print(f"train scene idces: {train_idces}, \nval scene idces: {val_idces}")
    
    if transform is None:
        transform = TRANSFORM

    ds_kwargs = dict(
        image_folder=image_folder, 
        annotation_file=annotation_file, 
        transform=transform,
        extra_info=extra_info,
    )
    ds_kwargs.update(kwargs)

    train_labeled_set = dataset_cls(scene_index=train_idces, **ds_kwargs)
    val_labeled_set   = dataset_cls(scene_index=val_idces,   **ds_kwargs)
    print(f"len(train_labeled_set)={len(train_labeled_set)}, len(val_labeled_set)={len(val_labeled_set)}")

    # it is always None, change code if you want
    # USE test_masktest_labeled_set
    test_labeled_set = None



    return train_labeled_set, val_labeled_set, test_labeled_set