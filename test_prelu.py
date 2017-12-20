from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils
import keras

# CNN Training parameters
batch_size = 10
nb_classes = 2
nb_epoch = 100

# Data Generation parameters
test_split = 0.2
dataset_size = 5000
patch_size = 32

# number of convolutional filters to use at each layer
nb_filters = [16, 32]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [7, 3]

# keras.layers.Conv3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), 
# 	activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', 
# 	kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

model = Sequential()
model.add(Conv3D(nb_filters[0],(nb_conv[0],nb_conv[0],nb_conv[0]),input_shape=(128, 128, 128, 1)))
act = keras.layers.advanced_activations.PReLU(weights=None, alpha_initializer="zero")
model.add(act)
model.add(MaxPooling3D(pool_size=(nb_pool[0], nb_pool[0], nb_pool[0])))
model.add(Dropout(0.5))
model.add(Conv3D(nb_filters[1],(nb_conv[1],nb_conv[1],nb_conv[1])))
model.add(MaxPooling3D(pool_size=(nb_pool[1], nb_pool[1], nb_pool[1])))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(16, activation='relu', kernel_initializer="normal"))
model.add(Dense(nb_classes, kernel_initializer="normal"))
model.add(Activation('softmax'))

print(keras.__version__)