# Usage:    python3 GAN.py train epochs
#           python3 GAN.py evaluate model_path

from tensorflow.keras.layers import UpSampling2D, Lambda, Input, Dense, Reshape, Conv2DTranspose, BatchNormalization, ReLU, Activation, Conv2D, Flatten, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Stop tensorflow from spamming me with info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

def make_generator():
    dropout = 0.4
    depth=64
    dim=7

    generator = Sequential()
    generator.add(Conv2D(depth, 3, 1, padding='same', input_shape=(25,25,4)))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(ReLU())
    generator.add(Dropout(dropout))
    #generator.add(UpSampling2D())
    generator.add(Conv2D(int(depth/2),5,1,padding='same'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(ReLU())
    generator.add(UpSampling2D())
    generator.add(Conv2D(int(depth/4),5,1,padding='same'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(ReLU())
    generator.add(Conv2D(int(depth/8),5,1,padding='same'))
    generator.add(BatchNormalization(momentum=0.9))
    generator.add(ReLU())
    generator.add(Conv2D(4,5,1,padding='same'))
    generator.add(Activation('sigmoid'))
    #generator.summary()

    return generator

def make_discriminator():
    dropout = 0.4
    depth = 8

    discriminator=Sequential()
    #discriminator.add(Input(shape=(28,28,1), name='image'))
    discriminator.add(Conv2D(depth,5,strides=1,padding='same',input_shape=(50,50,4)))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(dropout))

    discriminator.add(Conv2D(depth*2,5,strides=2,padding='same',input_shape=(50,50,4)))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(dropout))

    discriminator.add(Conv2D(depth*4,5,strides=2,padding='same',input_shape=(50,50,4)))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(dropout))

    discriminator.add(Conv2D(depth*8,5,strides=2,padding='same',input_shape=(50,50,4)))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(dropout))

    discriminator.add(Flatten())
    discriminator.add(Dense(1,activation='sigmoid'))
    #discriminator.summary()


    #Optimizer for discriminator (binary cross entropy)
    optimizer = optimizers.RMSprop(lr=0.0002, decay=6e-8)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return discriminator

def make_gan(generator, discriminator):
    # Make the weights in the discriminator not trainable
    discriminator.trainable = False

    GAN=Sequential()
    GAN.add(generator)
    GAN.add(discriminator)

    optimizer = optimizers.RMSprop(lr=0.0001, decay=3e-8)
    GAN.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return GAN

# Returns the score of the generator, calculated by taking the mean of the discriminator's prediction for each of the generator's generated images from the validation set
def score_generator(generator, discriminator, x_val):
    score = 0
    for i in range(x_val.shape[0]):
        gen_img = generator.predict(x_val[i].reshape((1, 25, 25, 4)))
        score += discriminator.predict(gen_img)

    return float(score / x_val.shape[0])

# Returns the score of the discriminator, calculated by adding 1/2 the mean of the discriminator's predictions for each real image to 1/2 the mean of 1 - the discriminator's predictions for each generated image
def score_discriminator(generator, discriminator, x_val, y_val):
    scoreReal = 0
    for i in range(y_val.shape[0]):
        scoreReal += discriminator.predict(y_val[i].reshape((1, 50, 50, 4)))
    scoreReal /= y_val.shape[0]

    scoreGenerated = 0
    for i in range(x_val.shape[0]):
        gen_img = generator.predict(x_val[i].reshape((1, 25, 25, 4)))
        scoreGenerated += 1 - discriminator.predict(gen_img)
    scoreGenerated /= x_val.shape[0]

    return float((scoreReal / 2) + (scoreGenerated / 2))

# Plots the scores of the generator and the discriminator for each epoch
def plot_scores(g_scores, d_scores):
    #x = [*range(0, len(g_scores))]
    plt.plot(g_scores, color='blue', label='Generator')
    plt.plot(d_scores, color='red', label='Discriminator')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Score')
    plt.title('Number of Epochs vs Score')
    plt.legend()
    plt.show()

def train_gan(generator, discriminator, GAN, downscaled_imgs, original_imgs, x_val, y_val, epochs):
    batch_size = 128

    generator_scores = []
    discriminator_scores = []

    #creating ground truth labels for discriminator
    valid = np.ones((batch_size))
    fake = np.zeros((batch_size))
    y = np.concatenate((valid,fake))

    #training procedure
    epoch = 0
    for i in range(int((original_imgs.shape[0]/batch_size)) * epochs):
        #sample random images from training set
        idx = np.random.randint(0, original_imgs.shape[0], batch_size)
        imgs_orig = original_imgs[idx]
        #sample random downscaled images
        idy = np.random.randint(0, downscaled_imgs.shape[0], batch_size)
        imgs_down = downscaled_imgs[idy]
        imgs_predicted = generator.predict(imgs_down)
        #create training set minibatch
        x = np.concatenate((imgs_orig, imgs_predicted))
        #train discriminator
        d_loss = discriminator.train_on_batch(x,y)
        #train generator (entire GAN)
        g_loss = GAN.train_on_batch(imgs_down, valid)

        # Every epoch...
        if i % (int(original_imgs.shape[0]/batch_size)) == 0:
            # Show losses and accuracies
            print(f'Epoch {epoch}:')
            epoch += 1
            print(f'discriminator loss: {d_loss[0]}    discriminator accuracy: {d_loss[1]}')
            print(f'generator loss: {d_loss[0]}    generator accuracy: {d_loss[1]}')
            print('=============================================================================')
            #print('{} d_loss: {}, g_loss{}'.format(i,d_loss,g_loss))

            # Calculate the scores of the generator and discriminator
            generator_scores.append(score_generator(generator, discriminator, x_val))
            discriminator_scores.append(score_discriminator(generator, discriminator, x_val, y_val))

        # Every 100 epochs, store a generated image
        if epoch % 100 == 0:
            img_down = downscaled_imgs[0].reshape((1, 25, 25, 4))
            new_img = generator.predict(img_down)
            image = new_img[0, :, :, :]
            filename = "pred_sample_" + str(epoch) + ".png"
            plt.imsave(filename, image, cmap='bone')

    plot_scores(generator_scores, discriminator_scores)
    return generator, discriminator, GAN

def main():
    if (len(sys.argv) != 3 or (sys.argv[1] != 'train' and sys.argv[1] != 'evaluate')):
        print("Usage:\tpython3 GAN.py train epochs\n\tpython3 GAN.py evaluate model_path")
        return

    # Load or create the models
    if sys.argv[1] == 'evaluate':
        generator = load_model(sys.argv[2] + '/generator_model')
        discriminator = load_model(sys.argv[2] + '/discriminator_model')
        GAN = load_model(sys.argv[2]+ '/GAN_model')
        print('Loading models...done.')

        # Read the data
        print('Reading data...', end='', flush=True)
        x_train, x_val, x_test, y_train, y_val, y_test = read_data()
        print('done.')

    elif sys.argv[1] == 'train':
        start = time.time()
        print('Creating models...', end='', flush=True)
        generator = make_generator()
        discriminator = make_discriminator()
        GAN = make_gan(generator, discriminator)
        print('done.')

        # Read the data
        print('Reading data...', end='', flush=True)
        x_train, x_val, x_test, y_train, y_val, y_test = read_data()
        print('done.')

        # Train the models, then save them
        epochs = int(sys.argv[2])
        generator, discriminator, GAN = train_gan(generator, discriminator, GAN, x_train, y_train, x_val, y_val, epochs)
        generator.save('GAN_models/generator_model')
        discriminator.save('GAN_models/discriminator_model')
        GAN.save('GAN_models/GAN_model')
        print("Training Time: %s seconds" % (time.time() - start))

    # Plotting predicted images
    n=10
    plt.figure(figsize=(20, 2))
    for i in range(1,n):
        ax = plt.subplot(1, n, i)
        idy = np.random.randint(0, x_train.shape[0], 1)
        img_down = x_train[idy]
        new_img=generator.predict(img_down)
        image = new_img[0, :, :, :]
        #image = np.reshape(image, [50, 50, 4])
        plt.imshow(image, cmap='bone')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


if __name__=='__main__':
    main()
