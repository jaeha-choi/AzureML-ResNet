#!/bin/python

import numpy as np
import tensorflow as tf
from tensorflow import keras

DATA_DIR = "tiny-imagenet-200"
CLASS_NO = 200
IMAGE_SZ = (64, 64)
BATCH_SZ = 64
EPOCH_NO = 50

# Load dataset
dataset = 

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
