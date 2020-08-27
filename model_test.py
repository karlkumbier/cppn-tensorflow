import numpy as np
import tensorflow as tf

from keras.layers import Dense, Input, Lambda, Reshape
from keras.models import Model
from keras.backend import tile, expand_dims
from keras.initializers import RandomNormal
from keras.activations import tanh


