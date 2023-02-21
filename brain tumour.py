
#import modules and pytorch libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
import torch.nn.functional as F
print("Libraries imported - ready to use PyTorch", torch.__version__)

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

##############################################################################
batch_size = 100
#importing training and test dataset
training_data = torchvision.datasets.mydata('path/to/root/', 
    train=False, 
    download=True, 
    transform=ToTensor())

test_data = torchvision.datasets.mydata('path/to/root/', 
    train=False, 
    download=True, 
    transform=ToTensor())

#split training data into training and validation
valid_ratio = 0.9 
n_training_data = int(len(training_data)) * valid_ratio
n_validation_data = len(training_data) - n_training_data

training_data, validation_data = data.random_split(training_data, n_validation_data, n_validation_data)


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
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 2 Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Dense layers
        self.fc1 = nn.Linear(64*64*64, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(p=0.20)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Add dropout layer here to prevent overfitting
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x
    
device = "cpu"
if (torch.cuda.is_available()):
    # if GPU available, use GPU in CNN training.
    device = "cuda"

model = CNN(num_classes = 2).to(device) # Define the final CNN as 'model'
summary(model, (1, 256, 256)) # get CNN architecture summary for our model
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