import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import os
import balance_data
import sys
import time

#function for reading in data for training/validation/testing
#data pulled from cropped folders
def read_data():
    #get all files
    downscaled_files = os.listdir("downscaled_cropped_images")
    original_files = os.listdir("original_cropped_images")

    #read in data
    x = []
    y = []
    for file in downscaled_files:
        im = plt.imread("downscaled_cropped_images/" + file)
        x.append(np.array(im))

    for file in original_files:
        im = plt.imread("original_cropped_images/" + file)
        y.append(np.array(im))

    #get balanced split of data
    x_training, x_validation, x_testing, y_training, y_validation, y_testing = balance_data.balance_data(x, y)

    return x_training, x_validation, x_testing, y_training, y_validation, y_testing

#showcase CNN model results on first 3 testing images
def CNN_results(x_testing, y_testing):
    #load model based on loss
    model1 = keras.models.load_model("CNN_models/CNN_MSE_model")
    model2 = keras.models.load_model("CNN_models/CNN_SSIM_model", custom_objects={ 'SSIM': SSIM })
    model3 = keras.models.load_model("CNN_models/CNN_SSIM_and_MSE_model", custom_objects={ 'SSIM_and_MSE': SSIM_and_MSE })
    model4 = keras.models.load_model("CNN_models/CNN_SSIM_then_MSE_model")

    fig = plt.figure(figsize=(8,8))

    for j in range(3):
        sample = x_testing[j].reshape((1, 25, 25, 4))
        y = y_testing[j].reshape((1, 50, 50, 4))

        pred1 = model1.predict(sample)
        pred2 = model2.predict(sample)
        pred3 = model3.predict(sample)
        pred4 = model4.predict(sample)

        for i in range(1,7):
            axes = fig.add_subplot(3,6, i + (j * 6))
            axes.set_xticks([])
            axes.set_yticks([])
            if i == 1:
                if j == 2:
                    axes.set_xlabel('X', fontsize=20)
                plt.imshow(sample[0], cmap="bone")
            elif i == 2:
                if j == 2:
                    axes.set_xlabel('MSE', fontsize=20)
                plt.imshow(pred1[0], cmap="bone")
            elif i == 3:
                if j == 2:
                    axes.set_xlabel('SSIM', fontsize=20)
                plt.imshow(pred2[0], cmap="bone")
            elif i == 4:
                if j == 2:
                    axes.set_xlabel('SSIM and MSE', fontsize=20)
                plt.imshow(pred3[0], cmap="bone")
            elif i == 5:
                if j == 2:
                    axes.set_xlabel('SSIM then MSE', fontsize=20)
                plt.imshow(pred4[0], cmap="bone")
            elif i == 6:
                if j == 2:
                    axes.set_xlabel('Y', fontsize=20)
                plt.imshow(y[0], cmap="bone")


    plt.show()

#showcase GAN model results on first 3 testing images
def GAN_results(x_testing, y_testing):
    #load model based on loss
    model1 = keras.models.load_model("GAN_models/generator_model")

    fig = plt.figure(figsize=(8,8))

    for j in range(3):
        sample = x_testing[j].reshape((1, 25, 25, 4))
        y = y_testing[j].reshape((1, 50, 50, 4))

        pred1 = model1.predict(sample)

        for i in range(1,4):
            axes = fig.add_subplot(3,3, i + (j * 3))
            axes.set_xticks([])
            axes.set_yticks([])
            if i == 1:
                if j == 2:
                    axes.set_xlabel('X', fontsize=20)
                plt.imshow(sample[0], cmap="bone")
            elif i == 2:
                if j == 2:
                    axes.set_xlabel('Prediction', fontsize=20)
                plt.imshow(pred1[0], cmap="bone")
            elif i == 3:
                if j == 2:
                    axes.set_xlabel('Y', fontsize=20)
                plt.imshow(y[0], cmap="bone")

    plt.show()


#custom loss function using both MSE and SSIM where each is weighted
def SSIM_and_MSE(y_true, y_pred):
    return 0.4 * keras.losses.mse(y_true, y_pred) + 0.6 * (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1)))

#SSIM loss function
def SSIM(y_true, y_pred):
    return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1)))

if __name__ == "__main__":
    #get data
    x_training, x_validation, x_testing, y_training, y_validation, y_testing = read_data()

    if len(sys.argv) == 2:
        if sys.argv[1] == "CNN":
            CNN_results(x_testing, y_testing)
        else:
            GAN_results(x_testing, y_testing)

    else:
        print("Usage: python3 plot_visual_predictions.py <CNN | GAN>")