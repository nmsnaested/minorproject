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

def test(model, test_loader, criterion, epoch, batch_log, device):
    correct_cnt = 0
    total_loss = 0.0

    model.eval()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(1)
            correct = (predicted == labels).long().sum().item()
            correct_cnt += correct
            
        avg_loss = total_loss / len(test_loader.dataset)
        tot_acc = 100. * correct_cnt / len(test_loader.dataset)
        print('Testing Epoch: {} \tAverage loss: {:.5f}\tAverage acc: {:.0f}%'.format(
            epoch, avg_loss, tot_acc ))
        
    return avg_loss, tot_acc