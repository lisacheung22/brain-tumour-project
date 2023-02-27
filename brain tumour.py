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
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
print("Libraries imported - ready to use PyTorch", torch.__version__)
##############################################################################
#importing training and test dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale (if needed)
    transforms.Resize((256, 256)),  # Resize the image to (256, 256) pixels
    transforms.ToTensor(),  # Convert to a tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize the pixel values to [-1, 1]
])

# Load the images from the two folders
# firstly create new folder named 'data' containing yes_output and no_output files
image_set = ImageFolder(root='.../Br35H/data', transform=transform)

# Define the ratio for each set
train_ratio = 0.8  # 80% for training
val_ratio = 0.1    # 10% for validation
test_ratio = 0.1   # 10% for testing

# Calculate the lengths of each set
train_len = int(len(image_set) * train_ratio)
val_len = int(len(image_set) * val_ratio)
test_len = len(image_set) - train_len - val_len

# Split the dataset using random_split
train_set, val_set, test_set = random_split(image_set, [train_len, val_len, test_len])

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
##############################################################################
# Here I display one random image from image_set to check
number = random.randint(0, 2999)
image, label = image_set[number]
image = np.transpose(image, (1, 2, 0))

# Display the image in grayscale
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()
print(f"Image_No: {number}")
##############################################################################
# Display 5 random images from train_loader in original form
# To confirm we have successfully loaded images ready for training
def unnormalize(img):
    img = img / 2 + 0.5  # unnormalize the pixel values
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))  # transpose the dimensions
    return img

images, labels = next(iter(train_loader))

# Visualize the first 5 images and their labels
for i in range(5):
    plt.imshow(unnormalize(images[i]), cmap='gray')
    plt.title('Label: {}'.format(labels[i]))
    plt.show()
##############################################################################
#define neural network class (CNN model architecture)
CNN = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    
    nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    
    nn.Flatten(),
    
    nn.Linear(64*64*64, 256),
    nn.ReLU(),
    nn.Dropout(p=0.20),
    
    nn.Linear(256, 1),
    nn.Sigmoid()
)

device = "cpu"
if torch.cuda.is_available():
    # if GPU available, use GPU in CNN training.
    device = "cuda"

model = CNN.to(device) # Define the final CNN as 'model'
print("The model will be running on", device, "device\n") 

# get model layer summary
print(model)
summary(CNN, (1, 256, 256))
##############################################################################
# training