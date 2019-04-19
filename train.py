# imports
import os
import keras
import shutil
import numpy as np
import pandas as pd
from keras import losses
from keras.callbacks import *
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# building model
inception = keras.applications.inception_v3.InceptionV3(include_top=True, 
                                       weights=None, 
                                       input_tensor=None, 
                                       input_shape=(1024,1024,3), 
                                       pooling=None, 
                                       classes=2,)

# image loading + preproccessing
datagen = ImageDataGenerator() ###### EDIT THIS LATER IF OVERFITTING BECOMES AN ISSUE

# certain hyperparameters
batch = 5

print("Loading training data...")
train_generator = datagen.flow_from_directory(
    directory= 'gci_data/train',
    target_size=(1024, 1024),
    color_mode="rgb",
    batch_size=batch,
    class_mode='categorical',
    shuffle=True,)

print("Loading validation data...")
valid_generator = datagen.flow_from_directory(
    directory= 'gci_data/val',
    target_size=(1024, 1024),
    color_mode="rgb",
    batch_size=batch,
    class_mode='categorical',
    shuffle=True,)

print("Compiling model...")
inception.compile(optimizer='nadam', loss='categorical_crossentropy', metrics =['accuracy'])

# defining callbacks
TensorBoard = TensorBoard(log_dir='./inception_logs', batch_size=batch)
ModelCheckpoint = ModelCheckpoint(filepath='./inception_models/weights.{epoch:02d}.hdf5')

history = inception.fit_generator(generator=train_generator,
                    validation_data = valid_generator,
                    steps_per_epoch=1500,
                    epochs=15,
                    callbacks=[TensorBoard,ModelCheckpoint])


inception.save('inception_models/inception_model')
