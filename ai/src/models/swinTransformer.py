"""
This is a recreation of the Swin Transformer example found at
https://keras.io/examples/vision/swin_transformers/ that will be the
basis for our implementation of the Swin Transformer.
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import ops


num_classes = 100
input_shape = (32, 32, 3)

patch_size = (2, 2)
dropout_rate = 0.03
num_heads = 8
embed_dim = 64
num_mlp = 256
qkv_bias = True
window_size = 2
shift_size = 1
image_dim = 32

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 1e-3
batch_size = 128
num_epochs = 40
validation_split = 0.1
weight_decay = 0.0001
label_smoothing = 0.1