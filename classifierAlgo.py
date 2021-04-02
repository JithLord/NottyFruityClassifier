import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import ImageFile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt


class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
          if (logs.get('acc')>0.8):
            self.model.stop_training = True
            
ImageFile.LOAD_TRUNCATED_IMAGES = True
img_height = 150
img_width =  img_height
batch_size=32
callback = myCallback()
 
TRAINING_DIR = r"../input/fruit-recognition/train/train"
training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=(0.3,0.6),
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    fill_mode='nearest')
 

validation_datagen = ImageDataGenerator(rescale = 1./255)
 
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(img_height, img_width),
    class_mode='categorical',
    subset='training',
    shuffle=True,
    batch_size=5
)
 
validation_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(img_height, img_width),
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    batch_size=5
)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height,img_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='selu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='selu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(128, (3,3), activation='selu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Conv2D(32, (3,3), activation='selu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(33, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

history = model.fit(train_generator, validation_data=validation_generator, epochs=100, steps_per_epoch=200, verbose = 1, validation_steps=5, callbacks=[callback])

model.save("fruits.h5")

acc = history.history['acc']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training Accuracy')
plt.legend(loc=0)
plt.figure()
