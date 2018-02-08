#!/usr/bin/env python

from keras.models import load_model

import argparse
import csv
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt



def get_X_y(data_dir):
    data_file = os.path.join(data_dir, 'driving_log.csv')

    X, y = [], []
    with open(data_file) as csv_file:
        for row in csv.DictReader(csv_file):
            if row['Left'] == 'Left':
                continue

            if float(row['Speed']) < 10:
                continue

            center_img = os.path.join(data_dir, 'IMG', row['Center'].strip())
            steer = float(row['Steer'])

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


def _generator(batch_size, X, y):
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            sample_index = random.randint(0, len(X) - 1)

            image, sa = process_image(X[sample_index], y[sample_index])

            batch_X.append(image)
            batch_y.append(sa)

        yield np.array(batch_X), np.array(batch_y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot')

    parser.add_argument('--data', type=str, required=True, help='path to training data')
    parser.add_argument('--model', type=str, required=False, default=None, help='filename to load model from')

    args = parser.parse_args()

    batch_size = 128

    model = load_model(args.model)

    X, y = get_X_y(args.data)
    data_generator = _generator(batch_size, X, y)

    steps = len(X) // batch_size

    pred = model.predict_generator(data_generator, steps=steps)

    fig, ax = plt.subplots()
    ax.plot(y, 'b', label='Rzeczywiste wartosci', alpha=0.6)
    ax.plot(pred, 'r', label='Odpowiedzi modelu', alpha=0.6)

    plt.title('Porownanie modelu do rzeczywistosci')
    plt.xlabel('Kolejne klatki')
    plt.ylabel('Sterowanie')

    ax.legend()

    fig.savefig('model_steer.png')
    plt.show()
