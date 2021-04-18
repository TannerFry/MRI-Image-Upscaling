import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import os

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
    x = np.array(x)
    y = np.array(y)

    #split into training/validation/testing => ~80/10/10 split
    training = int(0.8 * x.shape[0])
    validation = int(0.1 * x.shape[0]) + training

    x_training = x[0:training]
    x_validation = x[training:validation]
    x_testing = x[validation:]
    y_training = y[0:training]
    y_validation = y[training:validation]
    y_testing = y[validation:]

    return x_training, x_validation, x_testing, y_training, y_validation, y_testing

#function for training a model
def train(model, x_training, x_validation, y_training, y_validation):


#function for running trained model against testing data
def evaluate(model, x_testing, y_testing):



if __name__ == "__main__":
    x_training, x_validation, x_testing, y_training, y_validation, y_testing = read_data()
