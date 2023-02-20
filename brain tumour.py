
#import modules
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim

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

#define neural network class (CNN model architecture)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 2 Convolution layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Dense layers
        self.fc1 = nn.Linear(64*64*64, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
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
        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x

model = CNN() # Define the final CNN as 'model'
summary(model, (1, 256, 256)) # get CNN architecture summary for our model