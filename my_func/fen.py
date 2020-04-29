"""MobileNet v2 models for Keras.

# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""


from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
from keras.utils.vis_utils import plot_model
from keras.layers import Concatenate

# from keras import backend as K
from tensorflow.keras import backend as K

import numpy as np
import time

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        alpha: Integer, width multiplier.
        r: Boolean, Whether to use the residuals.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Depth
    tchannel = K.int_shape(inputs)[channel_axis] * t
    # Width
    cchannel = int(filters * alpha)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        alpha: Integer, width multiplier.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.

    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

    return x


def stage_1(sz_input, sz_input2, alpha=1):
    oup_ch = 4
    inputs = Input(shape=(sz_input, sz_input2, 1))
    x = _conv_block(inputs, oup_ch, (1, 1), (1, 1))
    output = _inverted_residual_block(x, oup_ch, (3, 3), t=4, alpha=alpha, strides=1, n=1)
    return Model(inputs, output)

def stage_2(sz_input, sz_input2, alpha=1):
    sz_input3 = 4*81
    oup_ch = 256
    input_list = []
    for i in range(81):
        input_list.append(Input(shape=(sz_input, sz_input2, 4)))

    x = Concatenate(axis=3)(input_list)
    x = _inverted_residual_block(x, oup_ch, (3, 3), t=4, alpha=alpha, strides=1, n=1)
    output = Conv2D(1, (1, 1), padding='same')(x)
    return Model(input_list, output)

def FEN(sz_input, sz_input2, learning_rate, train=True):
    """FEN
    This function defines a MobileNetv2 architectures.

    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
        alpha: Integer, width multiplier, better in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4].

    # Returns
        FEN model.
    """
    
    """ 81 inputs"""
    input_list = []
    for i in range(81):
        input_list.append(Input(shape=(sz_input, sz_input2, 1)))

    """ stage_1"""
    feature_extraction_layer = stage_1(sz_input, sz_input2)

    feature_list = []
    for i in range(81):
        feature_list.append(feature_extraction_layer(input_list[i]))

    """ stage_2"""
    merge_feature = stage_2(sz_input, sz_input2)
    output = merge_feature(feature_list)

    model = Model(input_list, output)
    # plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)
    opt = Adam(lr=learning_rate)

    model.compile(optimizer=opt, loss='mae')
    
    return model


if __name__ == '__main__':
    size = 512
    model = FEN(size, size, 0.001)
    dum = np.zeros((1, size, size,1), dtype=np.float32)
    tmp_list = []
    for i in range(81):
        tmp_list.append(dum)

    dummy = model.predict(tmp_list, batch_size=1)
    print('benchmark runtime...')
    n = 5
    start = time.time()
    for _ in range(n):
        dummy = model.predict(tmp_list, batch_size=1)
    print('runtime:', (time.time()-start)/n)