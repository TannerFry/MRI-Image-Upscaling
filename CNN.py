import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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
    print(x[0])

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
    print()

#function for running trained model against testing data
def evaluate(model, x_testing, y_testing):
    print()


if __name__ == "__main__":
    x_training, x_validation, x_testing, y_training, y_validation, y_testing = read_data()

    #setup model
    model = Sequential()
    model.add(layers.Conv2D(128, 10, input_shape=(25,25,4), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, 5, input_shape=(25, 25, 4), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, 5, input_shape=(25, 25, 4), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, 3, input_shape=(25, 25, 4), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10000, activation='sigmoid'))  # output layer for gender

    model.add(layers.Reshape((50,50,4)))

    sgd = optimizers.Adam(lr=0.001)
    model.compile(loss='MSE', optimizer=sgd)
    history = model.fit(x_training, y_training, validation_data=(x_validation, y_validation),
                        batch_size=32, epochs=25, use_multiprocessing=True)

    #save / load model to keep from re-training
    #model.save("model_testing")
    #model = keras.models.load_model("model_testing")

    #get a sample and save the downscaled, upscaled, and predicted image for comparison
    sample = x_training[0].reshape((1,25,25,4))
    test = model.predict(sample)

    plt.imsave("x_training_sample_0.png", x_training[0], cmap='bone')
    plt.imsave("y_training_sample_0.png", y_training[0], cmap='bone')
    plt.imsave("pred_sample_0.png", test[0], cmap='bone')
