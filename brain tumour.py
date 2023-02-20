
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
    """Some Information about MyModule"""
    def __init__(self):
        super(CNN, self).__init__()
        #first layer 
        self.conv1 = nn.Conv1d(1, 16, groups=1, bias=True, kernel_size=5, padding=0, stride=1)
        self.relu1 = nn.relu(input, inplace=False)
        self.pool1 = nn.max_pool1d(input, kernel_size = 2, stride=None, padding=0)

        #second layer 
        self.conv2 = nn.Conv1d(in_channel = 16, out_channel = 32, groups=1, bias=True, kernel_size=5, padding=0, stride=1)
        self.relu2 = nn.relu(input, inplace=False)
        self.pool2 = nn.max_pool2d(input, kernel_size = 2, stride=None, padding=0)

        #initalise cross-entropy
        self.F = nn.linear(input, weight)

    def forward(self, x):
        #first layer
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        
        #second layer
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        
        # flatten
        out = out.view(out.size(0), -1)

        # Linear function 
        out = self.F(out)
    
model = CNN() # Define the final CNN as 'model'
