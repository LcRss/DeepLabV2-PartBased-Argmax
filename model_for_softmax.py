from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model, Input
from tensorflow.python.keras.utils.layer_utils import get_source_inputs


def simpleModel(input_tensor=None, input_shape=None):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        img_input = input_tensor

    x = Conv2D(filters=128, kernel_size=7, strides=2, use_bias=False, name='conv1', padding='same')(img_input)

    x = Activation('relu', name='conv1_relu')(x)

    x = Conv2D(filters=256, kernel_size=5, strides=1, use_bias=False, name='conv2', padding='same')(x)

    x = Activation('relu', name='conv2_relu')(x)

    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=2, use_bias=False, name='conv3', padding='same')(x)

    x = Activation('relu', name='conv3_relu')(x)

    # x = MaxPooling2D(strides=2, name='pool3')(x)

    x = Conv2D(filters=1024, kernel_size=3, strides=1, use_bias=False, name='conv4', padding='same')(x)

    x = Activation('relu', name='conv4_relu')(x)

    # x = MaxPooling2D(3,strides=2, name='pool4')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='simpleModel')

    return model

# model = simpleModel(input_shape=(256, 256, 21))
# print(model.summary())


# x = Conv2D(filters=128, kernel_size=7, strides=1, use_bias=False, name='conv1_simple', padding='same')(x)
#
# x = Activation('relu', name='conv1_relu_simple')(x)
#
# x = Conv2D(filters=256, kernel_size=5, strides=1, use_bias=False, name='conv2_simple', padding='same')(x)
#
# x = Activation('relu', name='conv2_relu_simple')(x)
#
# x = MaxPooling2D(strides=2, name='pool1_simple')(x)
#
# x = Conv2D(filters=512, kernel_size=3, strides=1, use_bias=False, name='conv3_simple', padding='same')(x)
#
# x = Activation('relu', name='conv3_relu_simple')(x)
#
# x = MaxPooling2D(strides=2, name='pool3_simple')(x)
#
# x = Conv2D(filters=1024, kernel_size=3, strides=1, use_bias=False, name='conv4_simple', padding='same')(x)
#
# x = Activation('relu', name='conv4_relu_simple')(x)
#
# x = MaxPooling2D(strides=2, name='pool4_simple')(x)
