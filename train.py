
import sys
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K

K.clear_session()

training_data = './data/training'
validation_data = './data/validation'

epochs = 20
width, height = 150, 150
batch_size = 32
steps = 1000
validation_steps = 300
filters_conv1 = 32
filters_conv2 = 64
filter_size1 = (3, 3)
filter_size2 = (2, 2)
pool_size = (2, 2)
classes = 4
learning_rate = 0.0004

training_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2, 
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_generator = training_datagen.flow_from_directory(
    training_data,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical'
)

cnn = Sequential()
cnn.add(Convolution2D(filters_conv1, filter_size1, padding="same", input_shape=(width, height, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Convolution2D(filters_conv2, filter_size2, padding="same"))
cnn.add(MaxPooling2D(pool_size=pool_size))

cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(classes, activation='softmax'))

cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=learning_rate),
            metrics=['accuracy'])

cnn.fit_generator(
    training_generator,
    steps_per_epoch=steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

target_dir = './model/'
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
cnn.save('./model/model.h5')
cnn.save_weights('./model/weights.h5')