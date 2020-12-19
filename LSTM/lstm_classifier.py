
from __future__ import print_function

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, Embedding
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from MLP.preprocessing import Preprocessor

batch_size = 256
num_classes = 40
epochs = 5
hidden_size = 1024

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = Preprocessor().extract_features()
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


model = Sequential()
model.add(Embedding(2500, 128, input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))
model.add(LSTM(hidden_size, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# optimizers: sgd, adam, rmsprop, adamax
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#checkpointer = ModelCheckpoint(filepath='models/sentiment_sequential.hdf5', verbose=1, save_best_only=True)
earlyStopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[earlyStopper])
score = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
