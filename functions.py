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


def filter_size(size, ratio, nratio): 
    filter_size = math.ceil(size * ratio ** (nratio - 1))
    if filter_size % 2 != size % 2:
        filter_size += 1
    return filter_size

def plot_figures(filename, name, mode, mean=False):
    '''
    :param mode: train, test or valid
    '''
    pickle_log = open(filename,'rb')
    params = pickle.load(pickle_log)
    nb_models = pickle.load(pickle_log)

    losses = []
    accs = []
    for ii in range(nb_models):
        dynamics = pickle.load(pickle_log)
        losses.append(dynamics["{}_loss".format(mode)])
        accs.append(dynamics["{}_acc".format(mode)])

    if mean:
        avg_loss = np.mean(np.array(losses), axis=0)
        avg_acc = np.mean(np.array(accs), axis=0)
        
        std_loss = np.std(np.array(losses), axis=0)
        std_acc = np.std(np.array(accs), axis=0)
        
        x = list(range(len(avg_loss)))

        plt.figure()
        plt.errorbar(x, avg_loss, yerr=std_loss)
        plt.title("Mean {} loss {}".format(mode, name)) 
        plt.xlabel("Epochs")
        plt.ylabel("Categorical cross entropy")
        plt.savefig("{}_loss_mean_{}.pdf".format(mode, name))

        plt.figure()
        plt.errorbar(x, avg_acc, yerr=std_acc)
        plt.title("Mean {} accuracy {}".format(mode, name))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig("{}_acc_mean_{}.pdf".format(mode, name))

    else: 
        plt.figure()
        for ii in range(len(losses)):
            plt.plot(losses[ii], label = "model {}".format(ii))
        plt.title("{} loss {}".format(mode, name))
        plt.xlabel("Epochs")
        plt.ylabel("Categorical cross entropy")
        plt.legend()
        plt.savefig("{}_losses_{}.pdf".format(mode, name))
        #plt.show()

        plt.figure()
        for ii in range(len(accs)):
            plt.plot(accs[ii], label = "model {}".format(ii))
        plt.title("{} accuracy {}".format(mode, name))
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("{}_accuracies_{}.pdf".format(mode, name))
        #plt.show()

    pickle_log.close()


def plot_gaussian(logfile, scales, figtype="pdf"):
    log = open(logfile, "rb")
    params = pickle.load(log)
    nb_models = pickle.load(log)

    glist = []
    while 1:
        try:
            glist.append(pickle.load(log))
        except (EOFError):
            break
    
    avg_test_losses = glist[-4]
    avg_test_accs = glist[-3]
    std_test_losses = glist[-2]
    std_test_accs = glist[-1]

    log.close()

    plt.figure()
    for m in range(nb_models): 
        plt.errorbar(scales, avg_test_losses[m], yerr=std_test_losses[m], label="model {}".format(m))
    plt.title("Mean Loss vs Test scale")
    plt.xlabel("Test scale")
    plt.ylabel("Categorical cross entropy")
    plt.legend()
    plt.savefig("avg_test_loss_gaussian_cifar.{}".format(figtype))

    plt.figure()
    for m in range(nb_models): 
        plt.errorbar(scales, avg_test_accs[m], yerr=std_test_accs[m], label="model {}".format(m))
    plt.title("Mean Accuracy vs Test scale")
    plt.xlabel("Test scale")
    plt.ylabel("Accuracy %")
    plt.legend()
    plt.savefig("avg_test_acc_gaussian_cifar.{}".format(figtype))

    plt.figure()
    for m in range(nb_models): 
        plt.errorbar(scales, [100-x for x in avg_test_accs[m]], yerr=std_test_accs[m], label="model {}".format(m))
    plt.title("Mean Error vs Test scale")
    plt.xlabel("Test scale")
    plt.ylabel("Error %")
    plt.legend()
    plt.savefig("avg_test_err_gaussian_cifar.{}".format(figtype))