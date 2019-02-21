
from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from MLP.preprocessing import Preprocessor

batch_size = 128
num_classes = 40
epochs = 5
MODE = "simple"

# the data, split between train and test sets
if MODE == "simple":
    (x_train, y_train), (x_test, y_test) = Preprocessor().extract_features()
elif MODE == "neighbors":
    (x_train, y_train), (x_test, y_test) = Preprocessor().extract_with_neighbor_features()
elif MODE == "derivatives":
    (x_train, y_train), (x_test, y_test) = Preprocessor().extract_with_derivatives()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1])
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(x_train.shape[1], 'features')

model = Sequential()
if MODE == "neighbors":
    model.add(Dense(512, activation='relu', input_shape=(x_train[0].shape[0], x_train[0].shape[1],)))
else:
    model.add(Dense(512, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))

if MODE == "neighbors":
    model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='models/sentiment_sequential.hdf5', verbose=1, save_best_only=True)
earlyStopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=7, verbose=0, mode='auto')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[checkpointer, earlyStopper])
score = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
