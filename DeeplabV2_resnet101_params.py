from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Input
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
import os


def ResNet(stack_fn,
           simple_fn=None,
           model_name='resnet',
           weights='imagenet',
           input_tensor_1=None,
           input_shape_1=None,
           input_tensor_2=None,
           input_shape_2=None,
           kernel_init='glorot_uniform',
           batch_norm=False,
           dil_rate_model=False,
           classes=21,
           mult_rate=1):
    """
    # Arguments
        stack_fn: a function that returns output tensor for the
            stacked residual blocks.
        model_name: string, model name.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if input_tensor_1 is None:
        img_input_1 = Input(shape=input_shape_1)
    else:
        img_input_1 = input_tensor_1

    if input_tensor_2 is None:
        img_input_2 = Input(shape=input_shape_2)
    else:
        img_input_2 = input_tensor_2

    bn_axis = 3

    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1', padding='same')(img_input_1)

    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)

    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = stack_fn(x)

    xx = simple_fn(img_input_2, kernel_init, batch_norm, mult_rate)

    # DeeplabV2

    if not dil_rate_model:

        y = concatenate([x, xx], axis=-1)

        b1 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu',
                    name='fc1_voc12_c0_combine', kernel_initializer=kernel_init,
                    padding='same')(y)

        b2 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(12, 12), activation='relu',
                    name='fc1_voc12_c1_combine', kernel_initializer=kernel_init,
                    padding='same')(y)

        b3 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(18, 18), activation='relu',
                    name='fc1_voc12_c2_combine', kernel_initializer=kernel_init,
                    padding='same')(y)

        b4 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(24, 24), activation='relu',
                    name='fc1_voc12_c3_combine', kernel_initializer=kernel_init,
                    padding='same')(y)

        s = Add()([b1, b2, b3, b4])

    else:

        b1 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu',
                    name='fc1_voc12_c0_combine', kernel_initializer=kernel_init,
                    padding='same')(x)

        xx1 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu',
                     name='simple_model_dilation_rate_6', kernel_initializer=kernel_init,
                     padding='same')(xx)

        y1 = concatenate([b1, xx1], axis=-1)

        b2 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(12, 12), activation='relu',
                    name='fc1_voc12_c1_combine', kernel_initializer=kernel_init,
                    padding='same')(x)

        xx2 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(12, 12), activation='relu',
                     name='simple_model_dilation_rate_12', kernel_initializer=kernel_init,
                     padding='same')(xx)

        y2 = concatenate([b2, xx2], axis=-1)

        b3 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(18, 18), activation='relu',
                    name='fc1_voc12_c2_combine', kernel_initializer=kernel_init,
                    padding='same')(x)

        xx3 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(18, 18), activation='relu',
                     name='simple_model_dilation_rate_18', kernel_initializer=kernel_init,
                     padding='same')(xx)

        y3 = concatenate([b3, xx3], axis=-1)

        b4 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(24, 24), activation='relu',
                    name='fc1_voc12_c3_combine', kernel_initializer=kernel_init,
                    padding='same')(x)

        xx4 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(24, 24), activation='relu',
                     name='simple_model_dilation_rate_24', kernel_initializer=kernel_init,
                     padding='same')(xx)

        # b1 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu',
        #             name='fc1_voc12_c0_combine', kernel_initializer= kernel_init,
        #             padding='same')(x)
        #
        # xx1 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu',
        #              name='simple_model_dilation_rate_6', kernel_initializer= kernel_init,
        #              padding='same')(xx)
        #
        # y1 = concatenate([b1, xx1], axis=-1)
        #
        # b2 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(12, 12), activation='relu',
        #             name='fc1_voc12_c1_combine', kernel_initializer= kernel_init,
        #             padding='same')(y1)
        #
        # xx2 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(12, 12), activation='relu',
        #              name='simple_model_dilation_rate_12', kernel_initializer= kernel_init,
        #              padding='same')(xx1)
        #
        # y2 = concatenate([b2, xx2], axis=-1)
        #
        # b3 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(18, 18), activation='relu',
        #             name='fc1_voc12_c2_combine', kernel_initializer= kernel_init,
        #             padding='same')(y2)
        #
        # xx3 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(18, 18), activation='relu',
        #              name='simple_model_dilation_rate_18', kernel_initializer= kernel_init,
        #              padding='same')(xx2)
        #
        # y3 = concatenate([b3, xx3], axis=-1)
        #
        # b4 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(24, 24), activation='relu',
        #             name='fc1_voc12_c3_combine', kernel_initializer= kernel_init,
        #             padding='same')(y3)
        #
        # xx4 = Conv2D(filters=classes, kernel_size=(3, 3), dilation_rate=(24, 24), activation='relu',
        #              name='simple_model_dilation_rate_24', kernel_initializer= kernel_init,
        #              padding='same')(xx3)

        y4 = concatenate([b4, xx4], axis=-1)

        s = Add()([y1, y2, y3, y4])

        s = Conv2D(filters=classes, kernel_size=1, use_bias=False, name='conv_output', padding='same',
                   kernel_initializer=kernel_init)(s)

    logits = Lambda(lambda xx: tf.image.resize_bilinear(xx, size=tf.shape(img_input_1)[1:3]), name='logits')(s)

    out = Activation('softmax')(logits)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor_1 is not None:
        inputs_1 = get_source_inputs(input_tensor_1)
    else:
        inputs_1 = img_input_1

    if input_tensor_2 is not None:
        inputs_2 = get_source_inputs(input_tensor_2)
    else:
        inputs_2 = img_input_2

    # Create model.
    model = Model([inputs_1, inputs_2], out, name=model_name)

    return model


def dilated_block1(x, filters, kernel_size=3, stride=1,
                   conv_shortcut=True, name=None, dilation_factor=(1, 1)):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """

    bn_axis = 3  # if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = Conv2D(4 * filters, 1, strides=stride, padding='same', use_bias=False,
                          name='res%s_branch1' % name)(x)
        shortcut = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='bn%s_branch1' % name)(shortcut)
    else:
        shortcut = x

    x = Conv2D(filters, 1, strides=stride, name='res%s_branch2a' % name, use_bias=False, padding='same')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn%s_branch2a' % name)(x)
    x = Activation('relu', name=name + '_1_relu')(x)
    #####
    x = Conv2D(filters, kernel_size, padding='same', use_bias=False, dilation_rate=dilation_factor,
               name='res%s_branch2b' % name)(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn%s_branch2b' % name)(x)
    x = Activation('relu', name=name + '_2_relu')(x)
    #####
    x = Conv2D(4 * filters, 1, name='res%s_branch2c' % name, padding='same', use_bias=False, )(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn%s_branch2c' % name)(x)
    #####
    x = Add(name='res%s' % name)([shortcut, x])
    x = Activation('relu', name='res%s_relu' % name)(x)
    return x


def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """

    bn_axis = 3

    if conv_shortcut is True:
        shortcut = Conv2D(4 * filters, 1, strides=stride, padding='same', use_bias=False,
                          name='res%s_branch1' % name)(x)
        shortcut = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='bn%s_branch1' % name)(shortcut)
    else:
        shortcut = x

    x = Conv2D(filters, 1, strides=stride, name='res%s_branch2a' % name, use_bias=False, padding='same')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn%s_branch2a' % name)(x)
    x = Activation('relu', name=name + '_1_relu')(x)
    #####
    x = Conv2D(filters, kernel_size, padding='same', use_bias=False,
               name='res%s_branch2b' % name)(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn%s_branch2b' % name)(x)
    x = Activation('relu', name=name + '_2_relu')(x)
    #####
    x = Conv2D(4 * filters, 1, name='res%s_branch2c' % name, padding='same', use_bias=False, )(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name='bn%s_branch2c' % name)(x)
    #####
    x = Add(name='res%s' % name)([shortcut, x])
    x = Activation('relu', name='res%s_relu' % name)(x)
    return x


def stack1(x, filters, blocks, name=None):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """

    if name == '2a':
        x = block1(x, filters, stride=1, name=name)
        x = block1(x, filters, stride=1, conv_shortcut=False, name='2b')
        x = block1(x, filters, stride=1, conv_shortcut=False, name='2c')
    elif name == '3a':
        x = block1(x, filters, stride=2, name=name)
        for i in range(1, blocks):
            x = block1(x, filters, stride=1, conv_shortcut=False, name='3b' + str(i))
    elif name == '4a':
        x = block1(x, filters, stride=1, name=name)
        for i in range(1, blocks):
            x = dilated_block1(x, filters, stride=1, conv_shortcut=False, name='4b' + str(i), dilation_factor=(2, 2))
    elif name == '5a':
        x = dilated_block1(x, filters, stride=1, name='5a', dilation_factor=(4, 4))
        x = dilated_block1(x, filters, stride=1, conv_shortcut=False, name='5b', dilation_factor=(4, 4))
        x = dilated_block1(x, filters, stride=1, conv_shortcut=False, name='5c', dilation_factor=(4, 4))

    return x


def simpleModel(x, kernel_init='glorot_uniform', batchNorm=False, mult_rate=1):
    x = Conv2D(filters=mult_rate * 128, kernel_size=7, strides=2, use_bias=False, name='conv1_simple', padding='same',
               kernel_initializer=kernel_init)(x)

    if batchNorm:
        x = BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1_BN_simple')(x)

    x = Activation('relu', name='conv1_relu_simple')(x)

    x = Conv2D(filters=mult_rate * 256, kernel_size=5, strides=1, use_bias=False, name='conv2_simple', padding='same',
               kernel_initializer=kernel_init)(x)

    if batchNorm:
        x = BatchNormalization(axis=3, epsilon=1.001e-5, name='conv2_BN_simple')(x)

    x = Activation('relu', name='conv2_relu_simple')(x)

    x = MaxPooling2D(3, strides=2, name='pool1_simple')(x)

    x = Conv2D(filters=mult_rate * 512, kernel_size=3, strides=2, use_bias=False, name='conv3_simple', padding='same',
               kernel_initializer=kernel_init)(x)

    if batchNorm:
        x = BatchNormalization(axis=3, epsilon=1.001e-5, name='conv3_BN_simple')(x)

    x = Activation('relu', name='conv3_relu_simple')(x)

    x = Conv2D(filters=mult_rate * 1024, kernel_size=3, strides=1, use_bias=False, name='conv4', padding='same',
               kernel_initializer=kernel_init)(x)

    if batchNorm:
        x = BatchNormalization(axis=3, epsilon=1.001e-5,
                               name='conv4_BN_simple')(x)

    x = Activation('relu', name='conv4_relu_simple')(x)

    return x


def ResNet101(
        weights='imagenet',
        input_tensor_1=None,
        input_shape_1=None,
        input_tensor_2=None,
        input_shape_2=None,
        mult_rate=1,
        kernel_init='glorot_uniform',
        batch_norm=False,
        dil_rate_model=False,
        classes=21):
    def stack_fn(x):
        x = stack1(x, 64, 3, name='2a')
        x = stack1(x, 128, 4, name='3a')
        x = stack1(x, 256, 23, name='4a')
        x = stack1(x, 512, 3, name='5a')

        return x

    def simple_fn(x, k_init, b_norm, m_rate):
        x = simpleModel(x, k_init, b_norm, m_rate)

        return x

    return ResNet(stack_fn=stack_fn, simple_fn=simple_fn, model_name='resnet101',
                  weights=weights,
                  input_tensor_1=input_tensor_1, input_shape_1=input_shape_1, input_tensor_2=input_tensor_2,
                  input_shape_2=input_shape_2, kernel_init=kernel_init, batch_norm=batch_norm, mult_rate=mult_rate,
                  dil_rate_model=dil_rate_model, classes=classes)
