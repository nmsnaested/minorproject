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
    def __init__(self, f_in=3, nb_classes=10):
        super().__init__()
        '''
        Standard convolution network
        :param f_in: number of input features 
        :param nb_classes: number of classes on output
        '''
        self.f_in = f_in
        self.nb_classes = nb_classes

        self.conv1 = nn.Conv2d(self.f_in, 36, 7, 2)
        self.conv2 = nn.Conv2d(36, 64, 5)
        self.conv3 = nn.Conv2d(64, 64, 5)
        self.conv4 = nn.Conv2d(64, 64, 5)
        self.conv5 = nn.Conv2d(64, 64, 5)
        self.conv6 = nn.Conv2d(64, 150, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(150, self.nb_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(-1, x.size(0))
        x = F.relu(self.fc(x))
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

        self.conv1 = ScaleConvolution(self.f_in, 36, 3, self.ratio, self.nratio, srange = 0, boundary_condition = "dirichlet", stride = 2)
        self.pool1 = ScalePool(self.ratio)
        elf.conv2 = ScaleConvolution(36, 64, 3, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet")
        self.pool2 = ScalePool(self.ratio)
        self.conv3 = ScaleConvolution(64, 64, 3, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet")
        self.pool3 = ScalePool(self.ratio)
        self.conv4 = ScaleConvolution(64, 64, 3, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet")
        self.pool4 = ScalePool(self.ratio)
        self.conv5 = ScaleConvolution(64, 64, 3, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet")
        self.pool5 = ScalePool(self.ratio)
        self.conv6 = ScaleConvolution(64, 150, 3, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet")
        self.pool6 = ScalePool(self.ratio)

        self.fc = nn.Linear(150, self.nb_classes, bias = True)

    def forward(self, x): 
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)

        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        x = F.relu(self.conv5(x))
        x = self.pool5(x)

        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        x = F.relu(self.conv6(x))
        x = self.pool6(x)# [batch,feature]

        x = self.fc(x)
        return x


class SiCNN_3big(nn.Module): 
    def __init__(self, f_in=1, size=5, ratio=2**(2/3), nratio=3, srange=1, padding=0, nb_classes=10): 
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

        self.conv1 = ScaleConvolution(self.f_in, int(96*2.2), self.size, self.ratio, self.nratio, srange = 0, boundary_condition = "dirichlet", padding=self.padding, stride = 2)
        self.conv2 = ScaleConvolution(int(96*2.2), int(96*2.2), self.size, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet", padding=self.padding)
        self.conv3 = ScaleConvolution(int(96*2.2), int(192*2.2), self.size, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet", padding=self.padding)
        self.pool = ScalePool(self.ratio)
        
        self.fc1 = nn.Linear(int(192*2.2), int(150*2.2), bias=True)
        self.fc2 = nn.Linear(int(150*2.2), self.nb_classes, bias=True)

    def forward(self, x): 
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x) # [batch,feature]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SiCNN_3(nn.Module): 
    def __init__(self, f_in=1, size=5, ratio=2**(2/3), nratio=3, srange=2, padding=0, nb_classes=10): 
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

        self.conv1 = ScaleConvolution(self.f_in, 96, self.size, self.ratio, self.nratio, srange = 0, boundary_condition = "dirichlet", padding=self.padding, stride = 2)
        self.conv2 = ScaleConvolution(96, 96, self.size, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet", padding=self.padding)
        self.conv3 = ScaleConvolution(96, 192, self.size, self.ratio, self.nratio, srange = self.srange, boundary_condition = "dirichlet", padding=self.padding)
        self.pool = ScalePool(self.ratio)
        
        self.fc1 = nn.Linear(192, 150, bias=True)
        self.fc2 = nn.Linear(150, self.nb_classes, bias=True)

    def forward(self, x): 
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1)  # [batch, sigma, feature, y, x]
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x) # [batch,feature]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SiAllCNN(nn.Module): 
    def __init__(self, f_in, ratio, nratio, nb_classes=10):
        super().__init__()
        '''
        Squale equivariant All convolutional netw
        '''
        self.f_in = f_in
        self.ratio = ratio 
        self.nratio = nratio
        self.nb_classes = nb_classes

        self.conv1 = ScaleConvolution(self.f_in, 96, 3, ratio=self.ratio, nratio=self.nratio, srange=0, boundary_condition="dirichlet", padding=1)
        self.conv2 = ScaleConvolution(96, 96, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.conv3 = ScaleConvolution(96, 96, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1, stride=2)
        self.conv4 = ScaleConvolution(96, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.conv5 = ScaleConvolution(192, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.conv6 = ScaleConvolution(192, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1, stride=2)
        self.conv7 = ScaleConvolution(192, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        
        self.weight8 = nn.Parameter(torch.empty(192, 192))
        nn.init.orthogonal_(self.weight8)
        self.weight9 = nn.Parameter(torch.empty(self.nb_classes, 192))
        nn.init.orthogonal_(self.weight9)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1) # [batch, sigma, feature, y, x]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        
        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight8, x))
        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight9, x))

        n_batch = x.size(0)
        n_ratio = x.size(1)
        n_features_in = x.size(2)
        x = x.view(n_batch, n_ratio, n_features_in, -1).mean(-1) # [batch, sigma, feature]
        factors = x.new_tensor([self.ratio ** (-2 * i) for i in range(n_ratio)])
        x = torch.einsum("zsf,s->zf", (x, factors))  # [batch, feature]
       
        return x


class miniSiAll(nn.Module): 
    def __init__(self, f_in, ratio, nratio, nb_classes=10):
        super().__init__()
        '''
        Smaller version of the squale equivariant All CNN 
        '''
        self.f_in = f_in
        self.ratio = ratio 
        self.nratio = nratio
        self.nb_classes = nb_classes

        self.conv1 = ScaleConvolution(self.f_in, 96, 3, ratio=self.ratio, nratio=self.nratio, srange=0, boundary_condition="dirichlet", padding=1)
        self.conv2 = ScaleConvolution(96, 96, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.conv3 = ScaleConvolution(96, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.conv4 = ScaleConvolution(192, 192, 3, self.ratio, self.nratio, srange=2, boundary_condition="dirichlet", padding=1)
        self.weight5 = nn.Parameter(torch.empty(self.nb_classes, 192))
        nn.init.orthogonal_(self.weight5)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, sigma, feature, y, x]
        x = x.repeat(1, self.nratio, 1, 1, 1) # [batch, sigma, feature, y, x]

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.einsum("ij,bsjyx->bsiyx", (self.weight5, x))

        n_batch = x.size(0)
        n_ratio = x.size(1)
        n_features_in = x.size(2)
        x = x.view(n_batch, n_ratio, n_features_in, -1).mean(-1) # [batch, sigma, feature]
        factors = x.new_tensor([self.ratio ** (-2 * i) for i in range(n_ratio)])
        x = torch.einsum("zsf,s->zf", (x, factors))  # [batch, feature]
       
        return x