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

#function for training a model
def train(model, x_training, x_validation, y_training, y_validation, loss):
    #early stop based on patience
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # training with MSE loss
    if loss == "MSE":
        model.compile(loss='MSE', optimizer=optimizers.Adam(lr=0.00001, decay=0.000001))
        history = model.fit(x_training, y_training, validation_data=(x_validation, y_validation),
                            batch_size=32, epochs=1000, use_multiprocessing=True, callbacks=[es])
        # save model
        model.save("CNN_models/CNN_MSE_model")

        # save loss history
        with open("CNN_models/CNN_MSE_model_history.txt", "w") as f:
            f.write(str(history.history))

    #trainig with SSIM loss
    elif loss == "SSIM":
        model.compile(loss=SSIM, optimizer=optimizers.Adam(lr=0.00001, decay=0.000001))
        history = model.fit(x_training, y_training, validation_data=(x_validation, y_validation),
                            batch_size=32, epochs=1000, use_multiprocessing=True, callbacks=[es])
        # save model
        model.save("CNN_models/CNN_SSIM_model")

        # save loss history
        with open("CNN_models/CNN_SSIM_model_history.txt", "w") as f:
            f.write(str(history.history))

    #train with SSIM and MSE together
    elif loss == "SSIM_and_MSE":
        model.compile(loss=SSIM_and_MSE, optimizer=optimizers.Adam(lr=0.00001, decay=0.000001))
        history = model.fit(x_training, y_training, validation_data=(x_validation, y_validation),
                            batch_size=32, epochs=1000, use_multiprocessing=True, callbacks=[es])
        # save model
        model.save("CNN_models/CNN_SSIM_and_MSE_model")

        # save loss history
        with open("CNN_models/CNN_SSIM_and_MSE_model_history.txt", "w") as f:
            f.write(str(history.history))

    # train with SSIM then with MSE
    else:
        model.compile(loss=SSIM, optimizer=optimizers.Adam(lr=0.00001, decay=0.000001))
        history_ssim = model.fit(x_training, y_training, validation_data=(x_validation, y_validation),
                            batch_size=32, epochs=1000, use_multiprocessing=True, callbacks=[es])

        model.compile(loss="MSE", optimizer=optimizers.Adam(lr=0.00001, decay=0.000001))
        history_mse = model.fit(x_training, y_training, validation_data=(x_validation, y_validation),
                            batch_size=32, epochs=1000, use_multiprocessing=True, callbacks=[es])
        # save model
        model.save("CNN_models/CNN_SSIM_then_MSE_model")

        # save loss history
        with open("CNN_models/CNN_SSIM_then_MSE_model_history.txt", "w") as f:
            f.write(str(history_ssim.history) + "\n")
            f.write(str(history_mse.history))

#function for running trained model against testing data
def evaluate(model_path, x_testing, y_testing, loss):
    #load model based on loss
    if loss == "MSE" or loss == "SSIM_then_MSE":
        model = keras.models.load_model(model_path)
    elif loss == "SSIM":
        model = keras.models.load_model(model_path, custom_objects={ 'SSIM': SSIM })
    else:
        model = keras.models.load_model(model_path, custom_objects={ 'SSIM_and_MSE': SSIM_and_MSE })

    eval_path = model_path + "_testing_predictions/"
    if not os.path.isdir(eval_path):
        os.mkdir(eval_path)

    #loop through and predict for all testing samples, save images to evaluation folder
    for i in range(len(x_testing)):
        sample = x_testing[i].reshape((1, 25, 25, 4))
        pred = model.predict(sample)
        plt.imsave(eval_path + "testing_sample_" + str(i) + ".png", pred[0], cmap='bone')

#custom loss function using both MSE and SSIM where each is weighted
def SSIM_and_MSE(y_true, y_pred):
    return 0.4 * keras.losses.mse(y_true, y_pred) + 0.6 * (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1)))

#SSIM loss function
def SSIM(y_true, y_pred):
    return (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1)))

#setup model and return
def setup_model():
    # setup model
    model = Sequential()
    model.add(layers.Conv2D(128, 10, input_shape=(25, 25, 4), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, 5, input_shape=(25, 25, 4), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, 5, input_shape=(25, 25, 4), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, 3, input_shape=(25, 25, 4), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10000, activation='sigmoid'))  # output layer for gender
    model.add(layers.Reshape((50, 50, 4)))

    return model

if __name__ == "__main__":
    #get data
    x_training, x_validation, x_testing, y_training, y_validation, y_testing = read_data()

    #find best hyper-paramaters
    #hyper_paramater_testing(model, x_training, x_validation, y_training, y_validation)

    #make sure 3 input args
    if len(sys.argv) == 3:
        #if first arg 1 is train, setup model for training based on loss setup given in arg 2
        if sys.argv[1] == "train":
            loss = sys.argv[2]
            model = setup_model()
            train(model, x_training, x_validation, y_training, y_validation, loss)

    elif len(sys.argv) == 4:
        #if arg 1 is evaluate, load model based on model path given in arg 2, store loss of model used
        if sys.argv[1] == "evaluate":
            model_path = sys.argv[2]
            loss = sys.argv[3]
            evaluate(model_path, x_testing, y_testing, loss)

    else:
        #model = keras.models.load_model("CNN_models\\CNN_SSIM_and_MSE_model")

        print("Usage:\tpython3 CNN.py train <MSE | SSIM | SSIM_and_MSE | SSIM_then_MSE>\n\tpython3 CNN.py evaluate model_path <MSE | SSIM | SSIM_and_MSE | SSIM_then_MSE>")




