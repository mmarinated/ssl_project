import cv2
from PIL import Image

import numpy as np
from torchvision import transforms


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class RandomGaussianBluring(object):
    
    def __init__(self, kernel_size, p=0.5):
        self.kernel_size = kernel_size
        self.p = p
        
    def __call__(self, sample):
        sample = np.array(sample)
        
        if np.random.uniform(0,1) < self.p:
            sigma = np.random.uniform(0.1, 2.0)
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(sample)