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


def plot_gaussian(logfile, scales, logtype, figtype="pdf", dataset="cifar"):
    log = open(logfile, "rb")
    
    glist = []
    while 1:
        try:
            glist.append(pickle.load(log))
        except (EOFError):
            break
    models = glist[1] #2nd element , 1st el = parameters
    all_test_accs = glist[-1]

    log.close()

    for _, name in enumerate(models): 
        print(all_test_accs[name]["std"])

    plt.figure()
    for _, name in enumerate(models): 
        plt.errorbar(scales, [100-x for x in all_test_accs[name]["avg"]], yerr=all_test_accs[name]["std"], label=name)
    plt.title("Mean Error vs Test scale")
    plt.xlabel("Test scale")
    plt.ylabel("Error %")
    plt.legend()
    plt.savefig("avg_test_err_gaussian_{}_{}.{}".format(dataset, logtype, figtype))

def plot_train_log(logfile, models_dict, nb_epochs, figtype="pdf", dataset="cifar"):
    #models = []
    train_losses = []
    train_accs = []
    for m in range(len(models_dict)):
        log = open("trained_model_{}_{}.pickle".format(m, logfile) , "rb")
        tmplist = []
        while 1:
            try:
                tmplist.append(pickle.load(log))
            except (EOFError):
                break
        #name = tmplist[0]
        #models.append(name)
        train_dict = tmplist[-1]  #{"train_loss": train_loss, "train_acc": train_acc}
        train_losses.append(train_dict["train_loss"])
        train_accs.append(train_dict["train_acc"])

    plt.figure()
    for i, name in enumerate(models_dict): 
        plt.plot(range(nb_epochs), train_losses[i], label=name)
    plt.title("Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross entropy")
    plt.legend()
    plt.savefig("train_loss_{}_{}.{}".format(dataset, logfile, figtype))

    plt.figure()
    for i, name in enumerate(models_dict): 
        plt.plot(range(nb_epochs), train_accs[i], label=name)
    plt.title("Train accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("train_acc_{}_{}.{}".format(dataset, logfile, figtype))

def plot_train_val(logfile, models_dict, nb_epochs, figtype="pdf", dataset="cifar"):
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for m in range(len(models_dict)):
        log = open("trained_model_{}_{}_{}.pickle".format(m, nb_epochs, logfile) , "rb")
        tmplist = []
        while 1:
            try:
                tmplist.append(pickle.load(log))
            except (EOFError):
                break
        train_dict = tmplist[-1]  #{"train_loss": train_loss, "train_acc": train_acc}
        train_losses.append(train_dict["train_loss"])
        train_accs.append(train_dict["train_acc"])
        val_losses.append(train_dict["val_loss"])
        val_accs.append(train_dict["val_acc"])   
    
    plt.figure()
    for m, name in enumerate(models_dict): 
        plt.plot(range(nb_epochs), train_losses[m], label=name)
    plt.title("Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross entropy")
    plt.legend()
    plt.savefig("train_loss_{}_{}.{}".format(dataset, logfile, figtype))

    plt.figure()
    for m, name in enumerate(models_dict): 
        plt.plot(range(nb_epochs), train_accs[m], label=name)
    plt.title("Train accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("train_acc_{}_{}.{}".format(dataset, logfile, figtype))

    plt.figure()
    for m, name in enumerate(models_dict): 
        plt.plot(range(nb_epochs), val_losses[m], label=name)
    plt.title("Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross entropy")
    plt.legend()
    plt.savefig("val_loss_{}_{}.{}".format(dataset, logfile, figtype))

    plt.figure()
    for m, name in enumerate(models_dict): 
        plt.plot(range(nb_epochs), train_accs[m], label=name)
    plt.title("Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("val_acc_{}_{}.{}".format(dataset, logfile, figtype))