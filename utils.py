import numpy as numpy

def to_np(x):
    return x.detach().data.squeeze().numpy()