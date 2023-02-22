# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:51:09 2023

@author: Emily
"""

# building a simple model

import torch  # for Pytorch tensors
from PIL import Image  # python imaging library for loading & manipulating images
import numpy as np


def img_to_tensor(img):
    # Loads the image using .open() from PIL
    # converts it to the RGB colour space
    img = Image.open('image_001.png').convert(
        'RGB')  # img is now a PIL image object

    # converts img to a numpy array then to a PyTorch tensor object
    tensor_img = torch.Tensor(np.array(img))

    # Check if the image is all black or all white
    if (tensor_img.min() == 0 and tensor_img.max() == 0):
        return "The image is all black."
    elif (tensor_img.min() == 255 and tensor_img.max() == 255):
        return "The image is all white."
    else:
        return "The image is not all black or all white."
