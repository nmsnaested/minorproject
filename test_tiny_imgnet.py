#pylint: disable=E1101
import os
import os.path
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle

import torch
import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scale_cnn.convolution import ScaleConvolution
from scale_cnn.pooling import ScalePool

from loaddataset import ImgNetDataset

from architectures import StdNet, kanazawa, SiCNN
from train import train
from test import test 
from functions import filter_size, plot_train_val



device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


nb_epochs=200
learning_rate = 0.001
batch_size = 128
batch_log = 70

log = open("tiny_imagenet_log.pickle", "wb")

transforms = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

root = "./tiny-imagenet-200"
train_set = ImgNetDataset(rootdir=root, mode="train", transforms=transforms)
valid_set = ImgNetDataset(rootdir=root, mode="val", transforms=transforms)

train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers=1, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size = batch_size, shuffle = True, num_workers=1, pin_memory=True)

criterion = nn.CrossEntropyLoss()

models = {
    "standard ConvNet": StdNet(),
    "SiCNN k=5, r=2^(1/3), n=6, D=0": SiCNN(f_in = 3, size = 5, ratio = 2**(1/3), nratio = 6, srange = 0),
    "SiCNN k=5, r=2^(1/3), n=6, D=4": SiCNN(f_in = 3, size = 5, ratio = 2**(1/3), nratio = 6, srange = 4),
    "SiCNN wide, k=5, r=2^(1/3), n=6, D=0": SiCNN(f_in = 3, size = 5, ratio = 2**(1/3), nratio = 6, srange = 0, factor = 2.94),
    "SiCNN k=5, r=2^(2/3), n=3, D=0": SiCNN(f_in = 3, size = 5, ratio = 2**(2/3), nratio = 3, srange = 0),
    "SiCNN k=5, r=2^(2/3), n=3, D=2": SiCNN(f_in = 3, size = 5, ratio = 2**(2/3), nratio = 3, srange = 2),
    "SiCNN wide, k=5, r=2^(2/3), n=3, D=0": SiCNN(f_in = 3, size = 5, ratio = 2**(2/3), nratio = 3, srange = 0, factor=2.2),
    "SiCNN k=13, r=2^(-1/3), n=6, D=4":SiCNN(f_in = 3, size = filter_size(5, 2**(1/3), 6), ratio = 2**(-1/3), nratio = 6, srange = 4),
    "SiCNN k=13, r=2^(-2/3), n=3, D=2": SiCNN(f_in = 3, size =filter_size(5, 2**(2/3), 3), ratio = 2**(-2/3), nratio = 3, srange = 2),
    #"Kanazawa model, r=2^(1/3), n=6, D=0": kanazawa(f_in = 3, ratio = 2**(1/3), nratio = 6, srange = 0)
}

pickle.dump(models, log)

for m, name in enumerate(models): 
    print("model {}: {}".format(m, name))
    if os.path.isfile("trained_model_{}_tiny_{}.pickle".format(m, nb_epochs)): 
        model = pickle.load(open("trained_model_{}_tiny_{}.pickle".format(m, nb_epochs), "rb"))    
        model.to(device)
        epoch=nb_epochs
    else:
        model = models[name]
        model.to(device)
        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        for epoch in range(1, nb_epochs + 1): 
            train_l, train_a = train(model, train_loader, learning_rate, criterion, epoch, batch_log, device) 
            train_l, train_a = test(model, train_loader, criterion, epoch, batch_log, device) 
            train_loss.append(train_l)
            train_acc.append(train_a)
            val_l, val_a = test(model, valid_loader, criterion, epoch, batch_log, device)
            val_loss.append(val_l)
            val_acc.append(val_a)
        model_log = open("trained_model_{}_tiny_{}.pickle".format(m, nb_epochs), "wb")
        pickle.dump(name, model_log)
        pickle.dump(model, model_log)
        pickle.dump({
            "train_loss": train_loss,
            "train_acc": train_acc, 
            "val_loss": val_loss, 
            "val_acc": val_acc
            }, model_log)
        model_log.close()

log.close()

plot_train_val("tiny", models, nb_epochs, "pdf", "imagenet")
