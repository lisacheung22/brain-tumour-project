
#import modules and pytorch libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
print("Libraries imported - ready to use PyTorch", torch.__version__)

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

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
image_set = ImageFolder(root='path...', transform=transform)


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
#retrieves our dataset’s features and labels one sample at a time
#images are stored in img_dir 
#labels are stored in annotations_file
class MyDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self, annotations_file, mydata, transform=None, target_transform=None):
        self.img_labels = pd.read.csv(annotations_file)
        self.mydata = mydata
        self.transform = transform
        self.target_transfom = target_transform
        super(MyDataset, self).__init__()

    def __len__(self):
        return len(self.img_labels)
    
    #finds images by index and coverts into tensor using read image and matches the corresponding label, 
    #calls the transform function if possible
    def __getitem__(self, index):
        img_path = os.path.join(self.mydata, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
#labelling our dataset 
labels_map = {
    0:"tumour", 
    1:"no tumour"
}

#gets dataset feature and labels one sample at a time
dataloader = torch.utils.DataLoader(training_data, batch_size = batch_size, shuffle=True)
dataloader = torch.utils.DataLoader(test_data, batch_size = batch_size, shuffle=False)
dataloader = torch.utils.DataLoader(validation_data, batch_size = batch_size, shuffle=False)

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
print(model)

##############################################################################
# Define training function, return loss value
def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)       
        # Reset the optimizer
        optimizer.zero_grad()
        # Push the data forward through the model layers
        output = model(data)        
        # Get the loss
        loss = loss_criteria(output, target)
        # Keep a running total
        train_loss += loss.item()      
        # Backpropagate
        loss.backward()
        optimizer.step()        
        # Print metrics so we see some progress
        print('\tTraining batch {} Loss: {:.6f}'.format(batch_idx + 1, loss.item()))
            
    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx+1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss
##############################################################################
# Define testing function, return loss value
def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)            
            # Get the predicted classes for this batch
            output = model(data)            
            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()            
            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target==predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    # return average loss for the epoch
    return avg_loss
##############################################################################
# Define loss functions and optimizer
loss_criteria = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training the CNN model with our dataset
# Track metrics in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 20 epochs
epochs = 20
print('Training on', device)
for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
##############################################################################
# plot loss values over training epochs
plt.figure(figsize=(15,15))
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()