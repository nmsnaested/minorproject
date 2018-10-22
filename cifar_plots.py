import math
import matplotlib
import matplotlib.pyplot as plt
import pickle
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update(pgf_with_rc_fonts)
matplotlib.rcParams.update({'font.size': 9.5})
"""
scales = [(1.0, 1.0), (0.9, 1.1), (0.8, 1.2), (0.6, 1.4), (0.5, 1.5), (0.4, 1.6), (0.3, 1.7)]

lists = []
infile = open('cifar_range_log.pickle', 'rb')
while 1:
    try:
        lists.append(pickle.load(infile))
    except (EOFError):
        break
infile.close()


std_test_accs = lists[-1]
avg_test_accs = lists[-2]
std_test_losses = lists[-3]
avg_test_losses = lists[-4]

lists_sr0 = []
infile = open('cifar_range_sr0_log.pickle', 'rb')
while 1:
    try:
        lists_sr0.append(pickle.load(infile))
    except (EOFError):
        break
infile.close()

std_test_accs_sr0 = lists_sr0[-1]
avg_test_accs_sr0 = lists_sr0[-2]
std_test_losses_sr0 = lists_sr0[-3]
avg_test_losses_sr0 = lists_sr0[-4]


plt.figure(figsize=(4.1, 3.5))
plt.errorbar([str(s) for s in scales], avg_test_losses_sr0, yerr=[s/math.sqrt(6) for s in std_test_losses_sr0], label="$\Delta=0$")
plt.errorbar([str(s) for s in scales], avg_test_losses, yerr=[s/math.sqrt(6) for s in std_test_losses], label="$\Delta=2$")
plt.title("Average loss vs Scale factor")
plt.xlabel("Scale range")
plt.ylabel("Categorical cross entropy")
plt.legend()
plt.savefig("cifar_test_loss_range_mean.pgf")

plt.figure(figsize=(4.1, 3.5))
plt.errorbar([str(s) for s in scales], avg_test_accs_sr0, yerr=[s/math.sqrt(6) for s in std_test_accs_sr0], label="$\Delta=0$")
plt.errorbar([str(s) for s in scales], avg_test_accs, yerr=[s/math.sqrt(6) for s in std_test_accs], label="$\Delta=2$")
plt.title("Average accuracy vs Scale factor")
plt.xlabel("Scale range")
plt.ylabel("Accuracy %")
plt.legend()
plt.savefig("cifar_test_acc_range_mean.pgf")

plt.figure(figsize=(4.1, 3.5))
plt.errorbar([str(s) for s in scales], [100-x for x in avg_test_accs_sr0], yerr=[s/math.sqrt(6) for s in std_test_accs_sr0], label="$\Delta=0$")
plt.errorbar([str(s) for s in scales], [100-x for x in avg_test_accs], yerr=[s/math.sqrt(6) for s in std_test_accs], label="$\Delta=2$")
plt.title("Average error vs Train/Test scales")
plt.xlabel("Train and Test scales")
plt.ylabel("Error %")
plt.legend()
plt.savefig("cifar_test_err_range_mean.pgf")

################################################################

scales = [0.40, 0.52, 0.64, 0.76, 0.88, 1.0, 1.12, 1.24, 1.36, 1.48, 1.60]

infile = open("cifar_gaussian_log.pickle", "rb")

glist = []
while 1:
    try:
        glist.append(pickle.load(infile))
    except (EOFError):
        break


for i, m in enumerate(range(6,-1,-1)):
    locals()['avg_test_loss_{0}'.format(m)] = glist[-(4+i*4)]
    locals()['avg_test_acc_{0}'.format(m)] = glist[-(3+i*4)]
    locals()['std_test_loss_{0}'.format(m)] = glist[-(2+i*4)]
    locals()['std_test_acc_{0}'.format(m)] = glist[-(1+i*4)]


infile.close()

plt.figure(figsize=(4.1, 3.5))
plt.errorbar(scales, avg_test_loss_0, yerr=std_test_loss_0, label="$k=5,\, r=2^{2/3},\, n=3,\, \Delta=0$")
plt.errorbar(scales, avg_test_loss_1, yerr=std_test_loss_1, label="$k=5,\, r=2^{2/3},\, n=3,\, \Delta=2$")
plt.errorbar(scales, avg_test_loss_2, yerr=std_test_loss_2, label="$k=13,\, r=2^{-2/3},\, n=3,\, \Delta=2$")
plt.errorbar(scales, avg_test_loss_3, yerr=std_test_loss_3, label="Wide, $k=5,\, r=2^{2/3},\, n=3,\, \Delta=0$")
plt.errorbar(scales, avg_test_loss_4, yerr=std_test_loss_4, label="$k=5,\, r=2^{1/3},\, n=6,\, \Delta=0$")
plt.errorbar(scales, avg_test_loss_5, yerr=std_test_loss_5, label="$k=5,\, r=2^{1/3},\, n=6,\, \Delta=2$")
plt.errorbar(scales, avg_test_loss_6, yerr=std_test_loss_6, label="$k=13,\, r=2^{-1/3},\, n=6,\, \Delta=2$")
plt.title("Mean Loss vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Categorical cross entropy")
lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("avg_test_loss_gaussian_cifar.pgf", bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure(figsize=(4.1, 3.5))
plt.errorbar(scales, avg_test_acc_0, yerr=std_test_acc_0, label="$k=5,\, r=2^{2/3},\, n=3,\, \Delta=0$")
plt.errorbar(scales, avg_test_acc_1, yerr=std_test_acc_1, label="$k=5,\, r=2^{2/3},\, n=3,\, \Delta=2$")
plt.errorbar(scales, avg_test_acc_2, yerr=std_test_acc_2, label="$k=13,\, r=2^{-2/3},\, n=3,\, \Delta=2$")
plt.errorbar(scales, avg_test_acc_3, yerr=std_test_acc_3, label="Wide, $k=5,\, r=2^{2/3},\, n=3,\, \Delta=0$")
plt.errorbar(scales, avg_test_acc_4, yerr=std_test_acc_4, label="$k=5,\, r=2^{1/3},\, n=6,\, \Delta=0$")
plt.errorbar(scales, avg_test_acc_5, yerr=std_test_acc_5, label="$k=5,\, r=2^{1/3},\, n=6,\, \Delta=2$")
plt.errorbar(scales, avg_test_acc_6, yerr=std_test_acc_6, label="$k=13,\, r=2^{-1/3},\, n=6,\, \Delta=2$")
plt.title("Mean Accuracy vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Accuracy %")
lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("avg_test_acc_gaussian_cifar.pgf", bbox_extra_artists=(lgd,), bbox_inches='tight')

for m in range(7): 
    locals()['avg_test_err_{0}'.format(m)] = [100-l for l in locals()['avg_test_acc_{0}'.format(m)]]

plt.figure(figsize=(4.1, 3.5))
plt.errorbar(scales, avg_test_err_0, yerr=std_test_acc_0, label="$k=5,\, r=2^{2/3},\, n=3,\, \Delta=0$")
plt.errorbar(scales, avg_test_err_1, yerr=std_test_acc_1, label="$k=5,\, r=2^{2/3},\, n=3,\, \Delta=2$")
plt.errorbar(scales, avg_test_err_2, yerr=std_test_acc_2, label="$k=13,\, r=2^{-2/3},\, n=3,\, \Delta=2$")
plt.errorbar(scales, avg_test_err_3, yerr=std_test_acc_3, label="Wide, $k=5,\, r=2^{2/3},\, n=3,\, \Delta=0$")
plt.errorbar(scales, avg_test_err_4, yerr=std_test_acc_4, label="$k=5,\, r=2^{1/3},\, n=6,\, \Delta=0$")
plt.errorbar(scales, avg_test_err_5, yerr=std_test_acc_5, label="$k=5,\, r=2^{1/3},\, n=6,\, \Delta=2$")
plt.errorbar(scales, avg_test_err_6, yerr=std_test_acc_6, label="$k=13,\, r=2^{-1/3},\, n=6,\, \Delta=2$")
plt.title("Mean Error vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Error %")
lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("avg_test_err_gaussian_cifar.pgf", bbox_extra_artists=(lgd,), bbox_inches='tight')
"""


scales = [0.40, 0.52, 0.64, 0.76, 0.88, 1.0, 1.12, 1.24, 1.36, 1.48, 1.60]

log = open("cifar_gaussian_log.pickle", "rb")

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

log_big = open("cifar_gaussian_log_kbig.pickle", "rb")
glist_big = []
while 1:
    try:
        glist_big.append(pickle.load(log_big))
    except (EOFError):
        break

avg_test_losses_big = glist_big[-4]
avg_test_accs_big = glist_big[-3]
std_test_losses_big = glist_big[-2]
std_test_accs_big = glist_big[-1]

print(len(avg_test_losses))
print(len(scales))

plt.figure(figsize=(4.1, 3.5))
plt.errorbar(scales, avg_test_losses[0], yerr=std_test_losses[0], label="$r=2^{2/3}, n=3, \Delta=0$")
plt.errorbar(scales, avg_test_losses[1], yerr=std_test_losses[1], label="$r=2^{2/3}, n=3, \Delta=2$")
plt.errorbar(scales, avg_test_losses_big[0], yerr=std_test_losses_big[0], label="Wide, $r=2^{2/3}, n=3, \Delta=0$")
plt.errorbar(scales, avg_test_losses[2], yerr=std_test_losses[2], label="$r=2^{1/3}, n=6, \Delta=0$")
plt.errorbar(scales, avg_test_losses[3], yerr=std_test_losses[3], label="$r=2^{1/3}, n=6, \Delta=2$")
plt.title("Average loss vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Categorical cross entropy")
lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("test_loss_gaussian_k_mnist.pgf", bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure(figsize=(4.1, 3.5))
plt.errorbar(scales, avg_test_accs[0], yerr=std_test_accs[0], label="$r=2^{2/3}, n=3, \Delta=0$")
plt.errorbar(scales, avg_test_accs[1], yerr=std_test_accs[1], label="$r=2^{2/3}, n=3, \Delta=2$")
plt.errorbar(scales, avg_test_accs_big[0], yerr=std_test_accs[0], label="Wide, $r=2^{2/3}, n=3, \Delta=0$")
plt.errorbar(scales, avg_test_accs[2], yerr=std_test_accs[2], label="$r=2^{1/3}, n=6, \Delta=0$")
plt.errorbar(scales, avg_test_accs[3], yerr=std_test_accs[3], label="$r=2^{1/3}, n=6, \Delta=2$")
plt.title("Average accuracy vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Accuracy %")
lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("test_acc_gaussian_k_mnist.pgf", bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.figure(figsize=(4.1, 3.5))
plt.errorbar(scales_k, kanazawa, label="Kanazawa")
plt.errorbar(scales_k, convnet, label="ConvNet")
plt.errorbar(scales, [100-x for x in avg_test_accs[0]], yerr=std_test_accs[0], label="$r=2^{2/3}, n=3, \Delta=0$")
plt.errorbar(scales, [100-x for x in avg_test_accs[1]], yerr=std_test_accs[1], label="$r=2^{2/3}, n=3, \Delta=2$")
plt.errorbar(scales, [100-x for x in avg_test_accs_big[0]], yerr=std_test_accs_big[0], label="Wide, $r=2^{2/3}, n=3, \Delta=0$")
plt.errorbar(scales, [100-x for x in avg_test_accs[2]], yerr=std_test_accs[2], label="$r=2^{1/3}, n=6, \Delta=0$")
plt.errorbar(scales, [100-x for x in avg_test_accs[3]], yerr=std_test_accs[3], label="$r=2^{1/3}, n=6, \Delta=2$")
plt.title("Average error vs Test scale")
plt.xlabel("Test scale")
plt.ylabel("Error %")
lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig("test_err_gaussian_k_mnist.pgf", bbox_extra_artists=(lgd,), bbox_inches='tight')
