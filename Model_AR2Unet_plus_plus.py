import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np


def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    # Adjust shortcut if the number of filters is different
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x


def r2block(x, filters):
    x = residual_block(x, filters)
    x = residual_block(x, filters)
    return x


def transconv_block(x, filters, kernel_size=(3, 3), strides=(2, 2)):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def AdaptiveR2UNetPP(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Encoder
    conv1 = r2block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = r2block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = r2block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Bridge
    conv4 = r2block(pool3, 512)

    # Decoder
    upconv3 = transconv_block(conv4, 256)
    concat3 = layers.concatenate([upconv3, conv3], axis=-1)
    upconv3 = r2block(concat3, 256)

    upconv2 = transconv_block(upconv3, 128)
    concat2 = layers.concatenate([upconv2, conv2], axis=-1)
    upconv2 = r2block(concat2, 128)

    upconv1 = transconv_block(upconv2, 64)
    concat1 = layers.concatenate([upconv1, conv1], axis=-1)
    upconv1 = r2block(concat1, 64)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(upconv1)

    model = keras.Model(inputs, outputs)

    return model


def Model_AR2Unet_plus_plus(Data, Target, sol=None):
    if sol is None:
        sol = [5, 5, 300]
    input_shape = (256, 256, 3)
    num_classes = 3
    Train_Temp = np.zeros((Data.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Data.shape[0]):
        Train_Temp[i, :] = np.resize(Data[i], (input_shape[0], input_shape[1], input_shape[2]))
    Train_X = Train_Temp.reshape(Train_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    Test_Temp = np.zeros((Target.shape[0], input_shape[0], input_shape[1], input_shape[2]))
    for i in range(Target.shape[0]):
        Test_Temp[i, :] = np.resize(Target[i], (input_shape[0], input_shape[1], input_shape[2]))
    Train_Y = Test_Temp.reshape(Test_Temp.shape[0], input_shape[0], input_shape[1], input_shape[2])

    model = AdaptiveR2UNetPP(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(Train_X, Train_Y, epochs=int(sol[1]), steps_per_epoch=int(sol[2]))
    pred = model.predict(Data)
    return pred

