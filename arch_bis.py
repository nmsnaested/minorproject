#pylint: disable=E1101
'''
Architectures used
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from scale_cnn.convolution import ScaleConvolution
from scale_cnn.pooling import ScalePool

class StdNet(nn.Module):
    def __init__(self, f_in=3, size=5, nb_classes=10):
        super().__init__()
        '''
        Standard convolution network
        :param f_in: number of input features 
        :param size: kernel size
        :param nb_classes: number of classes on output
        '''
        self.f_in = f_in
        self.size = size
        self.nb_classes = nb_classes
        #64
        self.conv1 = nn.Conv2d(self.f_in, 12, self.size)
        self.conv2 = nn.Conv2d(12, 21, self.size)
        self.conv3 = nn.Conv2d(21, 36, self.size)
        self.conv4 = nn.Conv2d(36, 36, self.size)
        self.conv5 = nn.Conv2d(36, 64, self.size)
        self.conv6 = nn.Conv2d(64, 64, self.size) #40
        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(64 * 20**2, self.nb_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 20**2)
        x = self.fc(x)
        return x

class kanazawa(nn.Module): 
    def __init__(self, f_in, ratio, nratio, srange=1, nb_classes=10): 
        super().__init__()
        '''
        Scale equivariant arch, based on architecture in Kanazawa's paper https://arxiv.org/abs/1412.5104
        selecting srange = 1 is equivalent to the paper
        '''
        self.f_in = f_in
        self.ratio = ratio 
        self.nratio = nratio
        self.srange = srange
        self.nb_classes = nb_classes

        self.conv1 = ScaleConvolution(self.f_in, 12, 3, self.ratio, self.nratio, srange = 0, boundary_condition = "dirichlet", stride = 2)
        self.conv2 = ScaleConvolution(12, 21, 3, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet")
        self.conv3 = ScaleConvolution(21, 36, 3, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet")
        self.conv4 = ScaleConvolution(36, 36, 3, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet")
        self.conv5 = ScaleConvolution(36, 64, 3, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet")
        self.conv6 = ScaleConvolution(64, 64, 3, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet")
        self.pool = ScalePool(self.ratio)

        self.fc = nn.Linear(64, self.nb_classes, bias = True)

    def forward(self, x): 
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        x = F.relu(self.conv5(x))
        x = self.pool(x)

        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        x = F.relu(self.conv6(x))
        x = self.pool(x)# [batch,feature]

        x = self.fc(x)
        return x

class SiCNN(nn.Module): 
    def __init__(self, f_in=1, size=5, ratio=2**(2/3), nratio=3, srange=2, padding=0, nb_classes=10, factor=1): 
        super().__init__()
        '''
        Scale equivariant arch with 3 convolutional layers
        '''
        self.f_in = f_in
        self.size = size
        self.ratio = ratio 
        self.nratio = nratio
        self.srange = srange
        self.padding = padding
        self.nb_classes = nb_classes

        self.conv1 = ScaleConvolution(self.f_in, int(factor*12), self.size, self.ratio, self.nratio, srange = 0, boundary_condition = "dirichlet", padding=self.padding, stride = 2)
        self.conv2 = ScaleConvolution(int(factor*12), int(factor*21), self.size, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet", padding=self.padding)
        self.conv3 = ScaleConvolution(int(factor*21), int(factor*36), self.size, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet", padding=self.padding)
        self.conv4 = ScaleConvolution(int(factor*36), int(factor*36), self.size, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet", padding=self.padding)
        self.conv5 = ScaleConvolution(int(factor*36), int(factor*64), self.size, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet", padding=self.padding)
        self.conv6 = ScaleConvolution(int(factor*64), int(factor*64), self.size, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet", padding=self.padding)
        self.pool = ScalePool(self.ratio)
        
        self.fc = nn.Linear(int(factor*64), self.nb_classes, bias=True)

    def forward(self, x): 
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x) # [batch,feature]
        x = self.fc(x)
        return x