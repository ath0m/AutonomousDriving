#!/usr/bin/env python

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, SpatialDropout2D
from keras.layers.convolutional import Conv2D as Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,EarlyStopping

import cv2
import numpy as np

import random
import os
import csv
import argparse


def _model():
    input_shape = (128, 128)

    use_adadelta = True,
    learning_rate = 0.01
    W_l2 = 0.0001

    model = Sequential()

    model.add(Convolution2D(16, 5, 5,
                            input_shape=input_shape,
                            init="he_normal",
                            activation='relu',
                            border_mode='same'))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(20, 5, 5,
                            init="he_normal",
                            activation='relu',
                            border_mode='same'))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(40, 3, 3,
                            init="he_normal",
                            activation='relu',
                            border_mode='same'))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(60, 3, 3,
                            init="he_normal",
                            activation='relu',
                            border_mode='same'))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(80, 2, 2,
                            init="he_normal",
                            activation='relu',
                            border_mode='same'))
    model.add(SpatialDropout2D(0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 2, 2,
                            init="he_normal",
                            activation='relu',
                            border_mode='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=2,
                    init='he_normal',
                    W_regularizer=l2(W_l2)))

    optimizer = ('adadelta' if use_adadelta else SGD(lr=learning_rate, momentum=0.9))

    model.compile(loss='mean_squared_error', optimizer=optimizer)

    return model


def model(shape, load=None):
    if load:
        return load_model(load)

    conv_layers, dense_layers = [32, 32, 64, 128], [1024, 512]

    model = Sequential()

    model.add(Convolution2D(32, (3, 3), activation='elu', input_shape=shape))
    model.add(MaxPooling2D())

    for cl in conv_layers:
        model.add(Convolution2D(cl, (3, 3), activation='elu'))
        model.add(MaxPooling2D())
    model.add(Flatten())

    for dl in dense_layers:
        model.add(Dense(dl, activation='elu'))
        model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer="adam")

    return model


def get_X_y(data_dir, train=True):
    data_file = os.path.join(data_dir, 'driving_log.csv')

    X, y = [], []
    steering_offset = 0.1
    with open(data_file) as csv_file:
        for row in csv.DictReader(csv_file):
            if row['Left'] == 'Left':
                continue

            if float(row['Speed']) < 10:
                continue

            left_img = os.path.join(data_dir, 'IMG', row['Left'].strip())
            center_img = os.path.join(data_dir, 'IMG', row['Center'].strip())
            right_img = os.path.join(data_dir, 'IMG', row['Right'].strip())

            steer = float(row['Steer'])

            if train:
                X += [left_img, center_img, right_img]
                y += [steer - steering_offset, steer, steer + steering_offset]
            else:
                X.append(center_img)
                y.append(steer)

        return X, y


def process_image(path, steering_angle, shape=(128, 128)):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape

    image = image[h // 2:]
    image = cv2.resize(image, shape)[:, :, None]

    image = (image / 255. - .5).astype(np.float32)
    return image, steering_angle


def _generator(batch_size, X, y, train=True):
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            sample_index = random.randint(0, len(X) - 1)

            image, sa = process_image(X[sample_index], y[sample_index])

            batch_X.append(image)
            batch_y.append(sa)

            if train:
                batch_X.append(cv2.flip(image, 1)[:, :, None])
                batch_y.append(-sa)

        yield np.array(batch_X), np.array(batch_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Agent training')
    parser.add_argument('--load', type=str, required=False, default=None, help='filename to load model from')
    parser.add_argument('--save', type=str, required=False, default=None, help='filename to save model to')
    parser.add_argument('data', type=str, help='path to training data')
    parser.add_argument('--validation', type=str, required=False, help='path to validation data')
    parser.add_argument('--epochs', type=int, required=False, default=10, help='number of epochs')
    args = parser.parse_args()

    batch_size = 128

    net = model((128, 128, 1), args.load)
    outfile = args.save if args.save else 'model.h5'

    train_X, train_y = get_X_y(args.data)
    train_generator = _generator(batch_size, train_X, train_y)

    steps = 2 * len(train_X) // batch_size
    callbacks = []

    val_generator, val_steps = None, None
    if args.validation:
        val_X, val_y = get_X_y(args.validation, False)

        val_generator = _generator(batch_size, val_X, val_y, False)
        val_steps = len(val_X) // batch_size
        callbacks.append(ModelCheckpoint(outfile, monitor='val_loss', verbose=1, save_best_only=True))
        callbacks.append(EarlyStopping(monitor='val_loss', patience=20))
    else:
        callbacks.append(ModelCheckpoint(outfile, monitor='loss', verbose=1, save_best_only=True))

    net.fit_generator(train_generator, steps, epochs=args.epochs,
                      validation_data=val_generator, validation_steps=val_steps, callbacks=callbacks)
