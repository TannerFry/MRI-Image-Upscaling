import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
import keras_contrib.backend as KC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import os
import losses

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
    es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # training with SSIM loss
    model.compile(loss=custom_loss, optimizer=optimizers.Adam(lr=0.00001, decay=0.000001))
    history = model.fit(x_training, y_training, validation_data=(x_validation, y_validation),
                        batch_size=32, epochs=1000, use_multiprocessing=True, callbacks=[es])


    model.compile(loss=custom_loss, optimizer=optimizers.Adam(lr=0.00001, decay=0.000001))
    history = model.fit(x_training, y_training, validation_data=(x_validation, y_validation),
                        batch_size=32, epochs=1000, use_multiprocessing=True, callbacks=[es])


    # history = model.fit(x_training, y_training, validation_data=(x_validation, y_validation),
    #                     batch_size=32, epochs=100, use_multiprocessing=True)

    # for e in range(1000):
    #     history = model.fit(x_training, y_training, validation_data=(x_validation, y_validation),
    #                         batch_size=32, epochs=1, use_multiprocessing=True)
    #     #if e % 5 == 0:
    #     sample = x_training[0].reshape((1, 25, 25, 4))
    #     test = model.predict(sample)
    #     plt.imsave("pred_training_sample_0_0.png", test[0], cmap='bone')

    # sample = x_testing[1].reshape((1, 25, 25, 4))
    # test = model.predict(sample)
    # plt.imsave("x_test_sample_0_1.png", x_testing[1], cmap='bone')
    # plt.imsave("y_test_sample_0_1.png", y_testing[1], cmap='bone')
    # plt.imsave("pred_sample_0_1.png", test[0], cmap='bone')
    model.save("model")

    # sample = x_testing[1].reshape((1, 25, 25, 4))
    # test = model.predict(sample)
    # plt.imsave("x_test_sample_0_0.png", x_testing[1], cmap='bone')
    # plt.imsave("y_test_sample_0_0.png", y_testing[1], cmap='bone')
    # plt.imsave("pred_sample_0_0.png", test[0], cmap='bone')

    # training with MSR loss
    # model.compile(loss='MSE', optimizer=optimizers.Adam(lr=0.00001))
    # history = model.fit(x_training, y_training, validation_data=(x_validation, y_validation),
    #                     batch_size=32, epochs=1000, use_multiprocessing=True, callbacks=[es])

    # save / load model to keep from re-training
    # model.save("model_testing")
    # model = keras.models.load_model("model_2_losses_test")

    # get a sample and save the downscaled, upscaled, and predicted image for comparison
    # sample = x_testing[1].reshape((1, 25, 25, 4))
    # test = model.predict(sample)
    # plt.imsave("x_test_sample_0_1.png", x_testing[1], cmap='bone')
    # plt.imsave("y_test_sample_0_1.png", y_testing[1], cmap='bone')
    #plt.imsave("pred_sample_0_1.png", test[0], cmap='bone')

#function for running trained model against testing data
def evaluate(model, x_testing, y_testing):
    print()

#custom loss function
def custom_loss(y_true, y_pred):
    return 0.4 * keras.losses.mse(y_true, y_pred) + 0.6 * (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1)))

#test different hyperparamaters to find best setup
def hyper_paramater_testing(model, x_training, x_validation, y_training, y_validation):
    # Early stopping callback
    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001)

    best_val_loss = 100
    best_setup = ""

    learn_rate = [ 0.0001, 0.01]
    momentum = [0.0, 0.5]
    batch_size = [16, 64]
    loss = ['SSIM']
    with open("SGD_results.txt", "w") as f:
        for lr in learn_rate:
            for m in momentum:
                for bs in batch_size:
                    for l in loss:
                        # training with SSIM loss
                        if l == 'SSIM':
                            model.compile(loss=custom_loss, optimizer=optimizers.SGD(lr=lr, momentum=m))
                        else:
                            model.compile(loss=l, optimizer=optimizers.SGD(lr=lr, momentum=m))

                        history = model.fit(x_training, y_training, validation_data=(x_validation, y_validation),
                                            batch_size=bs, epochs=10000, use_multiprocessing=True, callbacks=[es])

                        if min(history.history['val_loss']) < best_val_loss:
                            best_setup = "SGD_model_" + str(lr) + "_" + str(m) + "_" + str(bs) + "_" +str(l)

                        f.write(str(history.history))
                        f.write("\n")

                    # save model
                    model.save("models/SGD_model_" + str(lr) + "_" + str(m) + "_" + str(bs) + "_" +str(l))

    with open("ADAM_results.txt", "w") as f:
        for lr in learn_rate:
            for bs in batch_size:
                for l in loss:
                    # training with SSIM loss
                    if loss == 'SSIM':
                        model.compile(loss=custom_loss, optimizer=optimizers.Adam(lr=lr))
                    else:
                        model.compile(loss=l, optimizer=optimizers.Adam(lr=lr))

                    history = model.fit(x_training, y_training, validation_data=(x_validation, y_validation),
                                        batch_size=bs, epochs=10000, use_multiprocessing=True, callbacks=[es])

                    if min(history.history['val_loss']) < best_val_loss:
                        best_setup = "ADAM_model_" + str(lr) + "_" + str(m) + "_" + str(bs) + "_" + str(l)

                    f.write(str(history.history))
                    f.write("\n")

                # save model
                model.save("models/ADAM_model_" + str(lr) + "_" + str(m) + "_" + str(bs) + "_" + str(l))

    print(best_setup)

if __name__ == "__main__":
    #get data
    x_training, x_validation, x_testing, y_training, y_validation, y_testing = read_data()

    #setup model
    model = Sequential()
    model.add(layers.Conv2D(128, 10, input_shape=(25, 25, 4), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, 5, input_shape=(25, 25, 4), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, 5, input_shape=(25, 25, 4), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, 3, input_shape=(25, 25, 4), activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10000, activation='sigmoid'))  # output layer for gender
    model.add(layers.Reshape((50, 50, 4)))

    #find best hyper-paramaters
    #hyper_paramater_testing(model, x_training, x_validation, y_training, y_validation)

    #train
    #train(model, x_training, x_validation, y_training, y_validation)

    model = keras.models.load_model("model", custom_objects={ 'custom_loss': custom_loss })

    #get a sample and save the downscaled, upscaled, and predicted image for comparison
    sample = x_testing[1].reshape((1, 25, 25, 4))
    test = model.predict(sample)
    plt.imsave("x_test_sample_0_1.png", x_testing[1], cmap='bone')
    plt.imsave("y_test_sample_0_1.png", y_testing[1], cmap='bone')
    plt.imsave("pred_sample_0_1.png", test[0], cmap='bone')
