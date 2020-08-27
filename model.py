import numpy as np
import tensorflow as tf

from keras.layers import Dense, Input, Lambda, Reshape
from keras.models import Model
from keras.backend import tile, expand_dims
from keras.initializers import RandomNormal
from keras.activations import tanh

class CPPN():
  def __init__(self, batch_size=1, z_dim = 32, c_dim = 1, scale = 8.0, net_size = 32):
    """

    Args:
    z_dim: how many dimensions of the latent space vector (R^z_dim)
    c_dim: 1 for mono, 3 for rgb.  dimension for output space.  you can modify code to do HSV rather than RGB.
    net_size: number of nodes for each fully connected layer of cppn
    scale: the bigger, the more zoomed out the picture becomes

    """

    self.batch_size = batch_size
    self.net_size = net_size
    self.scale = scale
    self.c_dim = c_dim
    self.z_dim = z_dim

    # builds the generator network
    self.G = self.generator()

  def _coordinates(self, x_dim = 32, y_dim = 32, scale = 1.0):
    '''
    calculates and returns a vector of x and y coordintes, and corresponding radius from the centre of image.
    '''
    n_points = x_dim * y_dim
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), self.batch_size).reshape(self.batch_size, n_points, 1)
    return x_mat, y_mat, r_mat

  def generator(self):

    net_size = self.net_size
    z_dim = self.z_dim
    c_dim = self.c_dim
    scale = self.scale
     
    x_input = Input((None, 1))
    y_input = Input((None, 1))
    r_input = Input((None, 1))

    z_input = Input((z_dim,))
    z_expand = Lambda(lambda x: expand_dims(z_input, axis=1))(z_input)
    z_scale = Lambda(lambda x: x * scale)(z_expand)

    x_unroll = Reshape((-1, 1), name='x_unroll')(x_input)
    y_unroll = Reshape((-1, 1), name='y_unroll')(y_input)
    r_unroll = Reshape((-1, 1), name='r_unroll')(r_input)

    normal_init = RandomNormal(mean=0.0, stddev=1.0)
    z_fc = DenseNorm(net_size)(z_scale)
    x_fc = Dense(net_size, kernel_initializer=normal_init)(x_unroll)
    y_fc = Dense(net_size, kernel_initializer=normal_init)(y_unroll)
    r_fc = Dense(net_size, kernel_initializer=normal_init)(r_unroll)
    U = Lambda(lambda x: x[0] + x[1] + x[2] + x[3])([z_fc, x_fc, y_fc, r_fc])

    H = Lambda(lambda x: tanh(x), name='tanh')(U)
    for i in range(3):
      H = DenseNorm(net_size, activation='tanh')(H)
    output = Dense(c_dim, activation='sigmoid')(H)

    model = Model(inputs=[z_input, x_input, y_input, r_input], outputs=output)
    return model

  def generate(self, z=None, x_dim = 1080, y_dim = 1080, scale = 8.0):
    """ Generate data by sampling from latent space.

    If z is not None, data for this point in latent space is
    generated. Otherwise, z is drawn from prior in latent
    space.
    """

    # TODO: how to handle size changes for initialized network 
    if z is None:
        z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim))

    if not hasattr(self, 'x_vec') or not hasattr(self, 'y_vec') or not hasattr(self, 'r_vec'):
        self.x_vec, self.y_vec, self.r_vec = self._coordinates(x_dim, y_dim, scale=scale)
    image = self.G.predict([z, self.x_vec, self.y_vec, self.r_vec])
    image = np.reshape(image, (x_dim, y_dim, self.c_dim))
    return image

def DenseNorm(net_size, activation=None, name=None):
    normal_init = RandomNormal(mean=0.0, stddev=1.0)
    dense_norm = Dense(net_size, 
                       kernel_initializer=normal_init,
                       bias_initializer=normal_init, 
                       activation=activation, 
                       name=name)

    return dense_norm

if __name__ == '__main__':

    from matplotlib import pyplot as plt
    cppn = CPPN(c_dim=3)
    out = cppn.generate(x_dim=4000, y_dim=4000, scale=40)

    print(out.shape)
    plt.imsave('test.png', out)

