#import modules and pytorch libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import random

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Dropout
#tf.config.set_visible_devices([], 'GPU')
print(tf.__version__)
##############################################################################
# reproducibility
random.seed(147)
np.random.seed(147)
torch.manual_seed(147)
tf.random.set_seed(147)
##############################################################################
# run this cell if your jupyter notebook kernel is died
#import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
##############################################################################