# %% [code]
# !pip install -U tensorflow_datasets
# !apt install -y fonts-nanum fonts-nanum-coding

import sys
import os
import math

import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras

import pathlib

STRING_CODEC = 'UTF-8'

TEXT_LEN = 64
TOKEN_LEN = 16
LATENT = 256

DATA_DIR = "tiny-imagenet-200"
CLASS_NO = 200
IMAGE_SZ = (64, 64)
BATCH_SZ = 64
EPOCH_NO = 50

# Any results you write to the current directory are saved as output.
print(tf.version.VERSION)

# Load dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    # Dataset must be organized into folders, check API below
    # https://keras.io/api/preprocessing/image/
    DATA_DIR,
    label_mode="int",
    color_mode="rgb",
    batch_size=BATCH_SZ,
    image_size=IMAGE_SZ,
    shuffle=True,
    interpolation="bilinear"
)

# Set model as ResNet50
model = tf.keras.applications.ResNet50(
    include_top=False,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=CLASS_NO
)

# Compile model, set metrics to display loss/accuracy
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)

history = model.fit(dataset, epochs=EPOCH_NO)
