import numpy as np
from Evaluation import evaluation
from keras import layers, models


def Model_GRU(X_train, y_train, X_test, Y_test, EP=None, Act=None, sol=None, batch=None):
    if sol is None:
        sol = [5, 5, 128, 5]
    if Act is None:
        Act = 'relu'
    if batch is None:
        batch = 4
    if EP is None:
        EP = 4

    input_shape = (32, 32, 3)
    time_steps = 5
    num_classes = Y_test.shape[-1]

    IMG_SIZE = 32
    Train_X = np.zeros((X_train.shape[0], time_steps, IMG_SIZE, IMG_SIZE, 3))
    for i in range(X_train.shape[0]):
        temp = np.resize(X_train[i], (time_steps, IMG_SIZE * IMG_SIZE, 3))
        Train_X[i] = np.reshape(temp, (time_steps, IMG_SIZE, IMG_SIZE, 3))

    Test_X = np.zeros((X_test.shape[0], time_steps, IMG_SIZE, IMG_SIZE, 3))
    for i in range(X_test.shape[0]):
        temp = np.resize(X_test[i], (time_steps, IMG_SIZE * IMG_SIZE, 3))
        Test_X[i] = np.reshape(temp, (time_steps, IMG_SIZE, IMG_SIZE, 3))

    model = models.Sequential()

    # GRU layer
    model.add(layers.InputLayer(input_shape=(time_steps,) + input_shape))
    # model.add(layers.InputLayer(input_shape=(time_steps + input_shape)))
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation=Act)))  # 'relu'
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation=Act)))  # 'relu'
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.GRU(int(sol[2]), return_sequences=True))
    model.add(layers.GRU(128))

    # Fully connected layers
    model.add(layers.Dense(128, activation=Act))  # 'relu'
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    model.fit(Train_X, y_train, epochs=5, batch_size=batch, validation_data=(Test_X, Y_test))
    # make predictions
    pred = model.predict(Test_X)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, Y_test)
    return Eval, pred
