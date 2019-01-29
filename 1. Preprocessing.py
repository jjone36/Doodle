
# This is first step of the entire script: data preprocessing part
# 1. loading dataset in parallel and
# 2. coverting strokes into image data
# 3. reshaping the size for inputting

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import ast
import cv2
import dask.bag as db

np.random.seed(36)

# list of animals
animals = ['ant', 'bat', 'bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow',
           'crab', 'crocodile', 'dog', 'dolphin', 'dragon', 'duck', 'elephant', 'fish',
           'flamingo', 'frog', 'giraffe', 'hedgehog', 'horse', 'kangaroo', 'lion',
           'lobster', 'monkey', 'mosquito', 'mouse', 'octopus', 'owl', 'panda',
           'parrot', 'penguin', 'pig', 'rabbit', 'raccoon', 'rhinoceros', 'scorpion',
           'sea turtle', 'shark', 'sheep', 'snail', 'snake', 'spider', 'squirrel',
           'swan', 'teddy-bear', 'tiger', 'whale', 'zebra']

# Setting
dir_path = '../input/train_simplified/'

im_size = 64
n_samples = 1000
n_class = len(animals)

# define a function converting drawing to image data
def draw_to_img(strokes, im_size = im_size):
    fig, ax = plt.subplots()                        # plot the drawing as we did above
    for x, y in strokes:
        ax.plot(x, -np.array(y), lw = 10)
    ax.axis('off')

    fig.canvas.draw()                               # update a figure that has been altered
    A = np.array(fig.canvas.renderer._renderer)     # converting them into array

    plt.close('all')
    plt.clf()

    A = (cv2.resize(A, (im_size, im_size)) / 255.)  # image resizing to uniform format

    return A[:, :, 3]

# Importing all samples
X_train = np.zeros((1, im_size, im_size))
y = []

for a in animals:
    print(a)
    filename = dir_path + a + '.csv'
    df = pd.read_csv(filename, usecols=['drawing', 'word'], nrows=n_samples)  # import the data in chunks
    df['drawing'] = df.drawing.map(ast.literal_eval)                          # convert strings into list
    X = df.drawing.values

    img_bag = db.from_sequence(X).map(draw_to_img)                            # covert strokes into array
    X = np.array(img_bag.compute())
    X_train = np.vstack((X_train, X))                                         # concatenate to get X_train

    y.append(df.word)


# The dimension of X_train
X_train.shape

# Drop the first layer
X_train = X_train[1:, :, :, :]
X_train.shape

# Encoding
y = pd.DataFrame(y)
y = pd.get_dummies(y)
y_train = np.array(y).transpose()

# Check the result
print("The input shape is {}".format(X_train.shape))
print("The output shape is {}".format(y_train.shape))

# Reshape X_train
X_train_2 = X_train.reshape((X_train.shape[0], im_size*im_size*3))

# Concatenate X_train and y_train
X_y_train = np.hstack((X_train_2, y_train))

# Random shuffle
np.random.shuffle(X_y_train)
a = im_size*im_size
cut = int(len(X_y_train) * .1)
X_val = X_y_train[:cut, :a]
y_val = X_y_train[:cut, a:]
X_train = X_y_train[cut:, :a]
y_train = X_y_train[cut:, a:]

# Reshape X_train back to (64, 64)
X_train = X_train.reshape((X_train.shape[0], im_size, im_size, 3))
X_val = X_val.reshape((X_val.shape[0], im_size, im_size, 3))

# Check the result
print("The input shape of train set is {}".format(X_train.shape))
print("The input shape of validation set is {}".format(X_val.shape))
print("The output shape of train set is {}".format(y_train.shape))
print("The output shape of validation set is {}".format(y_val.shape))
