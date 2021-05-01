import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import json
import pickle

def plot(histories, prev):
    # Plot all losses together
    colors = ['royalblue','red','springgreen','orange']
    names = ['MSE', 'SSIM & MSE Combined', 'SSIM', 'SSIM then MSE Sequential']
    i = 0
    for history in histories:
        loss = history['val_loss']
        epochs = range(1, len(loss)+1)
        plt.plot(epochs, loss, color=colors[i], label=names[i])
        i = i+1
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Number of Epochs vs Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('Plots/Validation/All_Loss_CNN_grid.png')
    plt.show()

    # Plot all individually
    i = 0
    for history in histories:
        loss = history['val_loss']
        epochs = range(1, len(loss)+1)
        plt.plot(epochs, loss, color=colors[i], label=names[i])
        plt.title('Number of Epochs vs Validation Loss ' + names[i])
        plt.xlabel('Number of Epochs')
        plt.ylabel('Valdation Loss')
        plt.legend()
        plt.grid()
        plt.savefig('Plots/Validation/' + names[i] + 'individual_grid.png')
        plt.show()
        i = i + 1

    # Plot Sequential with Distinction
    loss1 = prev[0]['val_loss']
    loss2 = prev[1]['val_loss']
    epochs1 = range(1, len(loss1)+1)
    epochs2 = range(len(loss1) +1, len(loss1) + len(loss2) + 1)
    plt.plot(epochs1, loss1, color=colors[3],label='SSIM')
    plt.plot(epochs2, loss2, color='red', label='MSE')

    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Number of Epochs vs Validation Loss SSIM then MSE Sequential')
    plt.legend()
    plt.grid()
    plt.savefig('Plots/Validation/' + names[3] + 'individual_distinct_grid.png')
    plt.show()

histories = []

for filename in os.listdir('Models'):

    with open('Models/' + filename, 'r') as f:
        file = f.read()
        json_acc = file.replace("'", "\"")
        history = json.loads(json_acc)
        if(filename == 'CNN_SSIM_then_MSE_model_history.txt'):
            prev = history
            h = history[0]
            h['val_loss'].extend(history[1]['val_loss'])
            history = h
           
            
        histories.append(history)
    
plot(histories, prev)