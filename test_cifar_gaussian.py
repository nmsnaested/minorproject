#pylint: disable=E1101
import os
import os.path
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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

#from architectures import StdNet, kanazawa, SiCNN
from arch_bis import StdNet, kanazawa, SiCNN
#from small_arch import StdNet, kanazawa, SiCNN
from train import train
from test import test 
from functions import filter_size, plot_gaussian 
from rescale import RandomRescale
import pickle

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

nb_epochs=200
learning_rate = 0.0001
batch_size = 128
batch_log = 70
repeats = 5

f_in = 3
size = 5

log = open("cifar_gaussian_log_bis.pickle", "wb")

parameters = {
    "epochs": nb_epochs,
    "learning rate": learning_rate,
    "batch size": batch_size,
    "repetitions": repeats
}
pickle.dump(parameters, log)

root = './cifardata' 
if not os.path.exists(root):
    os.mkdir(root)

train_transf = transforms.Compose([
                transforms.Resize(64), #RandomRescale(size = 40, scales = (1.0, 0.24), sampling = "normal"),  #apply scaling following a gaussian distribution, mean 1, std 0.24
                transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

train_set = datasets.CIFAR10(root=root, train=True, transform=train_transf, download=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

criterion = nn.CrossEntropyLoss()

scales = [0.28, 0.40, 0.52, 0.64, 0.76, 0.88, 1.0, 1.12, 1.24, 1.36, 1.48, 1.60, 1.72]

test_accs_dict = {}

models = {
    "standard ConvNet": StdNet(),
    "SiCNN k=5, r=2^(1/3), n=6, D=0": SiCNN(f_in = 3, size = 5, ratio = 2**(1/3), nratio = 6, srange = 0),
    "SiCNN k=5, r=2^(1/3), n=6, D=4": SiCNN(f_in = 3, size = 5, ratio = 2**(1/3), nratio = 6, srange = 4),
    "SiCNN wide, k=5, r=2^(1/3), n=6, D=0": SiCNN(f_in = 3, size = 5, ratio = 2**(1/3), nratio = 6, srange = 0, factor = 2.94),
    "SiCNN k=5, r=2^(2/3), n=3, D=0": SiCNN(f_in = 3, size = 5, ratio = 2**(2/3), nratio = 3, srange = 0),
    "SiCNN k=5, r=2^(2/3), n=3, D=2": SiCNN(f_in = 3, size = 5, ratio = 2**(2/3), nratio = 3, srange = 2),
    "SiCNN wide, k=5, r=2^(2/3), n=3, D=0": SiCNN(f_in = 3, size = 5, ratio = 2**(2/3), nratio = 3, srange = 0, factor=2.2),
    #"SiCNN k=13, r=2^(-1/3), n=6, D=4":SiCNN(f_in = 3, size = filter_size(5, 2**(1/3), 6), ratio = 2**(-1/3), nratio = 6, srange = 4),
    #"SiCNN k=13, r=2^(-2/3), n=3, D=2": SiCNN(f_in = 3, size =filter_size(5, 2**(2/3), 3), ratio = 2**(-2/3), nratio = 3, srange = 2),
    "Kanazawa model, r=2^(1/3), n=6, D=0": kanazawa(f_in = 3, ratio = 2**(1/3), nratio = 6, srange = 0)
}

pickle.dump(models, log)

for m, name in enumerate(models):
    print(" model {}: {}".format(m, name))
    if os.path.isfile("trained_model_{}_bis.pickle".format(m)): 
        model = pickle.load(open("trained_model_{}_bis.pickle".format(m), "rb"))
        model.to(device)
        epoch=200
    else:
        model = models[name]
        model.to(device)
        train_loss = []
        train_acc = []
        for epoch in range(1, nb_epochs + 1): 
            train_l, train_a = train(model, train_loader, learning_rate, criterion, epoch, batch_log, device) 
            train_l, train_a = test(model, train_loader, criterion, epoch, batch_log, device) 
            train_loss.append(train_l)
            train_acc.append(train_a)
        model_log = open("trained_model_{}_bis.pickle".format(m), "wb")
        pickle.dump(model, model_log)
        pickle.dump({"train_loss": train_loss, "train_acc": train_acc}, model_log)
        model_log.close()


    m_test_acc = []
    for ii in range(repeats):
        #lists of test acc for each scale with model m, trial ii
        test_acc = []
        for s in scales: 
            test_transf = transforms.Compose([
                                transforms.Resize(64), RandomRescale(size = 64, scales = (s, s), sampling = "uniform"),  #apply scaling uniformly 
                                transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])
            test_set = datasets.CIFAR10(root=root, train=False, transform=test_transf, download=True)
            test_loader = DataLoader(dataset=test_set, batch_size=batch_size,shuffle=False, num_workers=1, pin_memory=True)

            test_l, test_a = test(model, test_loader, criterion, epoch, batch_log, device)
            test_acc.append(test_a)
        m_test_acc.append(test_acc)
    avg_test_acc = np.mean(np.array(m_test_acc), axis=0)    
    std_test_acc = np.std(np.array(m_test_acc), axis=0)    

    pickle.dump(avg_test_acc, log)
    pickle.dump(std_test_acc, log)

    test_accs_dict[name] = {"avg": avg_test_acc, "std": std_test_acc}

pickle.dump(test_accs_dict, log)

log.close()

plot_gaussian("cifar_gaussian_log_bis.pickle", scales, "pdf")