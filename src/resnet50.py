#!/bin/python

import numpy as np
import tensorflow as tf
from tensorflow import keras

DATA_DIR = "tiny-imagenet-200"
CLASS_NO = 200
IMAGE_SZ = (64, 64)
BATCH_SZ = 1

# Load dataset
dataset = keras.preprocessing.image_dataset_from_directory(
  DATA_DIR, batch_size=BATCH_SZ, image_size=IMAGE_SZ)

# Set model as ResNet50
# To use pretrained weights, set weights='imagenet'
model = tf.keras.applications.ResNet50(
    include_top=True,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=CLASS_NO,
    **kwargs
)

model.summary()
