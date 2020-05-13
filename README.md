# Deep Learning Project

Final project on Bird's Eye Prediction for DL course

Papers and useful links:
- InfoVAE (https://arxiv.org/abs/1706.02262) <br>
- Understanding MMD: https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/ <br>
- MonoOccupancy (https://arxiv.org/pdf/1804.02176.pdf)
- UNet (https://arxiv.org/abs/1505.04597)
- Monocular Plan View Networks for Autonomous Driving: https://arxiv.org/pdf/1905.06937.pdf
- Review of papers on 3D object detection : https://towardsdatascience.com/monocular-3d-object-detection-in-autonomous-driving-2476a3c7f57e
- Inverse perspective mapping (IPM): from monocular images to Birds-eye-view (BEV) images


## Projections

Transfer labels from top down view to camera photos using projection geometry.<br>
https://kornia.github.io/<br>


## Road Layout Prediction

Refer to road_layout_prediction/ for code used to train and test road layout prediction models.

**Contents**:<br>
- road_layout_prediction.ipynb - Main notebook, Training & Evaluation<br>
- modelszoo.py - Model Architectures and Loss functions<br>
- simclr*, contrastive* - Code related to SimCLR implementaion for unsupervised training<br>

Parts of code sourced from:

- https://github.com/Chenyang-Lu/mono-semantic-occupancy<br>
- https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb<br>
- https://github.com/mdiephuis/SimCLR/blob/master/loss.py <br>
- https://github.com/guptv93/saycam-metric-learning/blob/master/data_util/simclr_transforms.py <br>

## Bounding Box Predictions

**Libraries used**<br>
- [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [FastAI](https://github.com/fastai/fastai)
- [Kornia](https://kornia.github.io/)
- [OpenCV](https://opencv.org/)

**Code**<br>
- Folder vehicle_layout_predictions
  - model_zoo.py - Model architectures used
  - pl_modules.py - Pytorch lightning modules
  - train.py - Example training code
- Notebook fastai_final_for_cars.ipynb
## Animations

### Predictions

![Predictions](animations/predictions.gif)

### Projection

![Projections](animations/BEV_projection_car_braking.gif)


