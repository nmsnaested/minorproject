#pylint: disable=E1101

import torch
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
#import time_logging

def train(model, train_loader, learning_rate, criterion, epoch, batch_log, device):
    running_loss = 0.0
    avg_loss = 0.0
    tot_acc = 0.0
    correct_cnt = 0
    
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    model.train()

    for idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        #t = time_logging.start()
        loss.backward()
        #time_logging.end("backward", t)
        optimizer.step()
        predicted = outputs.argmax(1)
        correct = (predicted == labels).long().sum().item()        
        correct_cnt += correct
        running_loss += loss.item()
        if idx % batch_log == (batch_log - 1):  
            tot_acc = 100. * correct_cnt / (len(images) * (idx + 1))
            avg_loss = running_loss / (idx + 1)
            print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f}\tAverage loss: {:.5f}\tAverage acc: {:.0f}%'.format(
                epoch, 
                (idx + 1) * len(images), 
                len(train_loader.dataset), 
                100. * (idx + 1) / len(train_loader), 
                loss.item(), 
                avg_loss, tot_acc
            ))

    #print(time_logging.text_statistics())
    return avg_loss, tot_acc