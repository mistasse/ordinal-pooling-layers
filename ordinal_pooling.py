import math
import numpy as np
import tensorflow as tf

from keras.models import Model
from keras.layers import Layer, Input, Lambda, Concatenate, MaxPool2D, GlobalMaxPool2D, AvgPool2D, GlobalAvgPool2D
from keras import backend as K

from keras.constraints import Constraint


class OrdinalPooling2D(Layer):
    """A layer that pools a 4D tensor by weighted sums of the elements in every pooling region.
    Weights are assigned depending on the rank of the element in the pooling region.
    """
    def __init__(self, pool_size=(2,2), stride=None, padding=0, initializer=None, **kwargs):
        self.pool_size = pool_size
        self.stride = pool_size if stride is None else stride
        self.padding = (padding, padding) if not isinstance(padding, tuple) else padding
        self.initializer = random_init if initializer is None else initializer
        super(OrdinalPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.ordinal_weights = self.add_weight(
            name="ordinal_weights",
            shape=[input_shape[-1], self.pool_size[0]*self.pool_size[1]],
            initializer=self.initializer,
            constraint=NormalizedPositiveWeights(),
            trainable=True,
        )
        super(OrdinalPooling2D, self).build(input_shape)

    def call(self, x):
        weights = self.ordinal_weights
        weights = weights * K.cast(K.greater_equal(weights, 0.), K.floatx())
        weights = weights / K.sum(weights, axis=-1, keepdims=True)
        padding = self.padding
        if padding != (0,0):
            x = K.spatial_2d_padding(x, padding=((padding[0], padding[0]), (padding[1], padding[1])))
        x = maxmin_sort_2d(x, pool_size=self.pool_size, stride=self.stride) # NhwCK
        x = x * weights[None,None,None,:,:]
        x = K.sum(x, axis=-1)
        return x

    def compute_output_shape(self, input_shape):
        n, h, w, c = input_shape
        H = math.ceil((h+self.padding[0]*2-(self.pool_size[0]-1)) / self.stride[0])
        W = math.ceil((w+self.padding[1]*2-(self.pool_size[1]-1)) / self.stride[1])
        return n, int(H), int(W), c


class GlobalOrdinalPooling2D(Layer):
    """A layer that globally ordinally pools any tensor of shape [N, H, W, C] into a tensor of shape [N, C]
    """

    def __init__(self, initializer=None, **kwargs):
        self.initializer = random_init if initializer is None else initializer
        super(GlobalOrdinalPooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        channels = input_shape[-1]
        self.ordinal_weights = self.add_weight(name="global_ordinal_weights",
                                               initializer=self.initializer,
                                               shape=[channels, input_shape[-2]*input_shape[-3]],
                                               trainable=True,
                                               constraint=NormalizedPositiveWeights())
        super(GlobalOrdinalPooling2D, self).build(input_shape)

    def call(self, x):
        x = K.permute_dimensions(x,(0,3,1,2))
        x = K.reshape(x,(K.shape(x)[0],K.shape(x)[1],K.shape(x)[2]*K.shape(x)[3]))
        x = tf.nn.top_k(x,k=K.shape(x)[-1]).values
        weights = self.ordinal_weights
        weights = weights * K.cast(K.greater_equal(weights, 0.), K.floatx())
        weights = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)   
        return K.sum(weights*x, axis=-1, keepdims=False)[:,None,None,:] # eventually weights[None,:]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, 1, input_shape[3])


def maxmin_sort(tensor):
    """Takes a tensor (or a tuple of tensors) and returns it sorted over the last
    dimension (using min-max swaps to avoid going on the CPU)"""
    if not isinstance(tensor, tuple):
        n = tensor.shape[-1]
        tensor = tuple(tensor[...,i] for i in range(n))
    else:
        n = len(tensor)
    for i in range(n):
        offset = i & 1
        tensor = sum([(K.maximum(tensor[j], tensor[j+1]),
                       K.minimum(tensor[j], tensor[j+1]))
                      if j <= n-2 else (tensor[j],)
                      for j in range(offset,n,2)], tensor[:offset])
    return K.stack(tensor, axis=-1)


def maxmin_sort_2d(x, pool_size=(2,2), stride=None):
    """Sort x with a spatial kernel (and stride), returns a new tensor with
    kx*ky as last sorted dimension"""
    n, h, w, c = x.shape.as_list()
    custom_stride = stride != pool_size
    kernel = pool_size
    if custom_stride:
        grid = []
        for i in range(0, h+1-kernel[0], stride[0]):
            grid.append([])
            for j in range(0, w+1-kernel[1], stride[1]):
                grid[-1].append(x[:,i:i+kernel[0], j:j+kernel[1],:])
        tensor = K.stack([K.stack(row, axis=2) for row in grid], axis=1) # NHkWkC
    else:
        tensor = K.reshape(x, [-1, h//kernel[0], kernel[0], w//kernel[1], kernel[1], c])
    return maxmin_sort(tuple(tensor[:,:,i,:,j,:]
                             for i in range(kernel[0])
                             for j in range(kernel[1])))


class NormalizedPositiveWeights(Constraint):
    def __call__(self, w):
        w = w * K.cast(K.greater_equal(w, 0.), K.floatx())  # Non negativity constraint
        w = w / K.sum(w, axis=-1, keepdims=True)            # Normalized weights constraint
        return w


def avg_pooling_init(shape):
    weights = np.ones(shape) / shape[1]
    return weights


def max_pooling_init(shape):
    weights = np.ones(shape) * np.array([1]+[0]*(shape[1]-1))[None]
    return weights


def random_init(shape):
    weights = np.random.rand(*shape)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return weights
