import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import RMSprop
from keras.callbacks import Callback, CSVLogger
from keras.utils import np_utils
from matplotlib import pyplot as pyplot
from sklearn.model_selection import train_test_split
import argparse
import numpy as np
from PIL import Image
import glob
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf

def main():
    #old_session = KTF.get_session()

    #session = tf.Session('')
    #KTF.set_session(session)
    #KTF.set_learning_phase(1)

    folder = ["nadeko", "kuroneko"]
    image_size = 28

    X = []
    Y = []
    for index, name in enumerate(folder):
        dir = "./" + name
        files = glob.glob(dir + "/*.jpg")
        for i, file in enumerate(files):
            image = Image.open(file)
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X.append(data)
            Y.append(index)
    X = np.array(X)
    Y = np.array(Y)
    X = X.astype('float32')
    X = X / 255.0
    Y = np_utils.to_categorical(Y, 2)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    model = Sequential()

    model.add(Conv2D(32, (3,3), padding='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

    tb_cb = keras.callbacks.TensorBoard(log_dir="./tflog/", histogram_freq=1)
    cbks = [tb_cb]

    batch_size = 4

    history = model.fit(X_train, y_train, 
                        batch_size=batch_size,
                        epochs=100, callbacks=cbks, validation_data=(X_test, y_test))

    print(model.evaluate(X_test, y_test))

    model.save("./hanaza.h5")

    #KTF.set_session(old_session)



if __name__ == "__main__":
    main()