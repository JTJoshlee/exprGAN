import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from pytorch_grad_cam import GradCAM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


class ResBlockReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.relu = nn.LeakyReLU(0.02)        
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        
        
    def forward(self, x):
        
        return self.relu(self.conv(x) + self.shortcut(x))
    
class ResBlockReLUBN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockReLUBN, self).__init__()
        # self.layers = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),  # First convolution layer
        #     nn.LeakyReLU(0.02),  # LeakyReLU activation
        #     nn.BatchNorm2d(out_channels),  # Batch normalization
        # )
        # self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.relu = nn.LeakyReLU(0.02)
        self.bn = nn.BatchNorm2d(out_channels)
        #if in_channels != out_channels:
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        #else:
            #self.shortcut = nn.Identity()

    def forward(self, x):
         return self.bn(self.relu(self.conv(x)+self.shortcut(x)))
        # return self.relu(self.bn(self.conv(x) + self.shortcut(x)))
    
class Encoder(nn.Module):
    def __init__(self, input_channels=3):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            ResBlockReLU(input_channels, 32),
            ResBlockReLUBN(32,64),
            ResBlockReLUBN(64,128),
            ResBlockReLUBN(128,256),
            ResBlockReLUBN(256,256)
        )

    def forward(self, x):       
        return self.layers(x)

class ExpressionClassifier(nn.Module):
    def __init__(self, input_size):
        super(ExpressionClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.Linear(256, 2),
            
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(input_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256,2)
        # )
    
    def forward(self, x):
        x_flat = x.reshape(x.size(0), -1)                
        x_flat = self.fc(x_flat)              
        return x_flat 
    
class EASNNetwork(nn.Module):
    def __init__(self, input_channels=3):
        super(EASNNetwork, self).__init__()
        
        self.encoder = Encoder(input_channels)
        
        # calculate the size of the flattened feature vector
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)
            encoded_output = self.encoder(dummy_input)
            encoded_size = encoded_output.view(1, -1).size(1)
        
        self.flatten_size = encoded_size
       
        self.ExpressionClassifier = ExpressionClassifier(encoded_size)
        #self.IdClassifier = IdClassifier(encoder_output_dim=512, flatten_size=encoded_size)
    def forward(self, x):
        
        x_encoded = self.encoder(x)        
        neutral = self.ExpressionClassifier(x_encoded)       
        return neutral, x_encoded

    def get_flatten_size(self):
        
        return self.flatten_size

class IdClassifier(nn.Module):
    def __init__(self, conv_channels=64):
        super(IdClassifier, self).__init__()

        
        self.encoder_ouput_dim = 256
        
        self.conv_bn_sigmoid = nn.Sequential(
            nn.Conv2d(self.encoder_ouput_dim, conv_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(conv_channels),            
            nn.Sigmoid()            
        )
    
        self.flatten_size = EASNNetwork().get_flatten_size()
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x, y):        
        xy_abs = torch.abs(x-y)
        features = self.conv_bn_sigmoid(xy_abs)               
        self.flatten_size = features.shape[1] * features.shape[2] * features.shape[3]
        self.fc1 = nn.Linear(self.flatten_size, 256).to(features.device)
        flattened = features.view(features.size(0), -1)
        out = self.fc1(flattened)
        out = self.fc2(out)
        out = self.fc3(out)
               
        
        return out






if __name__ == "__main__":
    model = EASNNetwork()
