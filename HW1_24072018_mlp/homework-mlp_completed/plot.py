
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_and_acc(loss_and_acc_dict):
    fig = plt.figure()
    max_epoch = len(loss_and_acc_dict.values()[0][0])
    stride = np.ceil(max_epoch / 10)
    
    max_loss = max([max(x[0]) for x in loss_and_acc_dict.values()]) + 0.1
    min_loss = max(0, min([min(x[0]) for x in loss_and_acc_dict.values()]) - 0.1)
    
    for name, loss_and_acc in loss_and_acc_dict.items():
        plt.plot(range(1, 1 + max_epoch), loss_and_acc[0], '-s', label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(range(0, max_epoch + 1, 2))
    plt.axis([0, max_epoch, min_loss, max_loss])
    plt.show()
    
    max_acc = min(1, max([max(x[1]) for x in loss_and_acc_dict.values()]) + 0.1)
    min_acc = max(0, min([min(x[1]) for x in loss_and_acc_dict.values()]) - 0.1)
    
    fig = plt.figure()
    for name, loss_and_acc in loss_and_acc_dict.items():
        plt.plot(range(1, 1 + max_epoch), loss_and_acc[1], '-s', label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(range(0, max_epoch + 1, 2))
    plt.axis([0, max_epoch, min_acc, max_acc])
    plt.legend()
    plt.show()
           