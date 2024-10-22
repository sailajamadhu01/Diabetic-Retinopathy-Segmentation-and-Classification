from Model_GRU import Model_GRU
import warnings
from keras import backend as K
import cv2 as cv
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
from ResidualAttentionNetwork import ResidualAttentionNetwork
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np


def Model_RAN_Feat(train_data, train_target, test_data, test_target, Batch_size=None):
    if Batch_size is None:
        Batch_size = 4

    IMAGE_WIDTH = 32
    IMAGE_HEIGHT = 32

    IMAGE_CHANNELS = 3
    IMAGE_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

    X_test = np.zeros((test_data.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), dtype=np.uint8)
    X_train = np.zeros((train_data.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS), dtype=np.uint8)

    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMAGE_WIDTH * IMAGE_HEIGHT, 3))
        X_train[i] = np.reshape(temp, (IMAGE_WIDTH, IMAGE_HEIGHT, 3))

    for i in range(test_data.shape[0]):
        temp = np.resize(train_data[i], (IMAGE_WIDTH * IMAGE_HEIGHT, 3))
        X_test[i] = np.reshape(temp, (IMAGE_WIDTH, IMAGE_HEIGHT, 3))

    batch_size = 32

    num_classes = test_target.shape[1]

    STEP_SIZE_TRAIN = len(train_data) // batch_size

    model_path = "/pylon5/cc5614p/deopha32/Saved_Models/cvd-model.h5"

    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True)
    csv_logger = CSVLogger("/pylon5/cc5614p/deopha32/Saved_Models/cvd-model-history.csv", append=True)

    callbacks = [checkpoint, csv_logger]

    # Model Training
    with tf.device('/gpu:0'):
        model = ResidualAttentionNetwork(
            input_shape=IMAGE_SHAPE,
            n_classes=num_classes,
            activation='softmax').build_model()

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        try:
            history = model.fit(X_train, steps_per_epoch=STEP_SIZE_TRAIN, verbose=0, callbacks=callbacks,
                                batch_size=Batch_size,
                                epochs=50, use_multiprocessing=True, workers=40)
        except:
            pass
        score = model.predict(X_test)
        # Eval = evaluation(score, test_target)

    Data = np.concatenate((X_train, X_test), axis=0)
    data = np.zeros((Data.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    for i in range(Data.shape[0]):
        temp = np.resize(Data[i], (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
        data[i] = np.reshape(temp, (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp], [out]) for out in outputs]  # evaluation functions
    layerNo = -2
    Feats = []
    for i in range(data.shape[0]):
        print(i, data.shape[0])
        test = data[i, :][np.newaxis, ...]
        layer_out = np.asarray(functors[layerNo]([test])).squeeze()  # [func([test]) for func in functors]
        Feats.append(layer_out)
    Feats = np.asarray(Feats)
    Feature = cv.resize(Feats, (Data.shape[1], Data.shape[0]))

    return Feature


def Model_RAN_GRU(Train_Data, Train_Target, Test_Data, Test_Target, EP=None):
    if EP is None:
        EP = 5
    Feat = Model_RAN_Feat(Train_Data, Train_Target, Test_Data, Test_Target, EP)
    Target = np.concatenate((Train_Target, Test_Target), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(Feat, Target, random_state=104, test_size=0.25, shuffle=True)

    Eval, pred = Model_GRU(X_train, y_train, X_test, y_test, EP)
    return Eval, pred


