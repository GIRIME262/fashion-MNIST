from keras.callbacks import ModelCheckpoint 
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import initializers
from fnn_helper import PlotLosses
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D

def get_model(p=0.3):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(28,28,1)))
    model.add(Convolution2D(filters=6, strides=1, kernel_size=6, padding='same', name='Conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Convolution2D(filters=12, strides=2, kernel_size=5, padding='same', name='Conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Convolution2D(filters=24, strides=2, kernel_size=4, padding='same', name='Conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(10, activation='softmax', name='OutputLayer'))
    return model

def get_model_2(p=0.3):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(28,28,1)))
    model.add(Convolution2D(filters=6, strides=1, kernel_size=3, padding='same', name='Conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Convolution2D(filters=12, strides=2, kernel_size=3, padding='same', name='Conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Convolution2D(filters=24, strides=2, kernel_size=3, padding='same', name='Conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(10, activation='softmax', name='OutputLayer'))
    return model

def get_model_3(p=0.3):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(28,28,1)))
    model.add(Convolution2D(filters=6, strides=1, kernel_size=3, padding='same', name='Conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Convolution2D(filters=12, strides=2, kernel_size=3, padding='same', name='Conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Convolution2D(filters=24, strides=2, kernel_size=3, padding='same', name='Conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Convolution2D(filters=48, strides=2, kernel_size=3, padding='same', name='Conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(p))
    model.add(Dense(10, activation='softmax', name='OutputLayer'))
    return model

def get_model_4():
    model3=Sequential()
    model3.add(BatchNormalization(input_shape=(28,28,1)))
    model3.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", name='Conv1',
              input_shape=(28,28,1)))
    model3.add(BatchNormalization())
    model3.add(Activation('relu'))
    model3.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", name='Conv2'))
    model3.add(BatchNormalization())
    model3.add(Activation('relu'))
    model3.add(MaxPooling2D(pool_size=(2, 2)))
    model3.add(Dropout(0.5))
    model3.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", name='Conv3'))
    model3.add(BatchNormalization())
    model3.add(Activation('relu'))
    model3.add(Conv2D(filters=256, kernel_size=(3, 3), padding="valid", name='Conv4'))
    model3.add(BatchNormalization())
    model3.add(Activation('relu'))
    model3.add(MaxPooling2D(pool_size=(3, 3)))
    model3.add(Dropout(0.5))
    model3.add(Flatten())
    model3.add(Dense(256))
    model3.add(LeakyReLU())
    model3.add(Dropout(0.5))
    model3.add(Dense(256))
    model3.add(LeakyReLU())
    model3.add(Dense(10, activation='softmax'))
    model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model3

def get_model_5():
    model3=Sequential()
    model3.add(BatchNormalization(input_shape=(28,28,1)))
    model3.add(Conv2D(filters=6, kernel_size=(3, 3), padding="same", name='Conv1',
              input_shape=(28,28,1)))
    model3.add(BatchNormalization())
    model3.add(Activation('relu'))
    model3.add(Conv2D(filters=12, kernel_size=(3, 3), padding="same", name='Conv2'))
    model3.add(BatchNormalization())
    model3.add(Activation('relu'))
    model3.add(MaxPooling2D(pool_size=(2, 2), name='MaxPool1'))
    model3.add(Dropout(0.5))
    model3.add(Conv2D(filters=24, kernel_size=(3, 3), padding="same", name='Conv3'))
    model3.add(BatchNormalization())
    model3.add(Activation('relu'))
    model3.add(Conv2D(filters=48, kernel_size=(3, 3), padding="valid", name='Conv4'))
    model3.add(BatchNormalization())
    model3.add(Activation('relu'))
    model3.add(MaxPooling2D(pool_size=(3, 3), name='MaxPool2'))
    model3.add(Dropout(0.5))
    model3.add(Flatten())
    model3.add(Dense(256))
    model3.add(LeakyReLU())
    model3.add(Dropout(0.5))
    model3.add(Dense(256))
    model3.add(LeakyReLU())
    model3.add(Dense(10, activation='softmax'))
    model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model3

def get_model_6():
    model3=Sequential()
    model3.add(BatchNormalization(input_shape=(28,28,1)))
    model3.add(Conv2D(filters=6, kernel_size=(3, 3), padding="same", name='Conv1',
              input_shape=(28,28,1)))
    model3.add(BatchNormalization())
    model3.add(Activation('relu'))
    model3.add(Conv2D(filters=12, kernel_size=(3, 3), padding="same", name='Conv2'))
    model3.add(BatchNormalization())
    model3.add(Activation('relu'))
    model3.add(MaxPooling2D(pool_size=(2, 2), name='MaxPool1'))
    #model3.add(Dropout(0.5))
    model3.add(Conv2D(filters=24, kernel_size=(3, 3), padding="same", name='Conv3'))
    model3.add(BatchNormalization())
    model3.add(Activation('relu'))
    #model3.add(Dropout(0.25))
    model3.add(Flatten())
    model3.add(Dense(256))
    model3.add(LeakyReLU())
    #model3.add(Dropout(0.25))
    model3.add(Dense(256))
    model3.add(LeakyReLU())
    model3.add(Dense(10, activation='softmax'))
    model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model3