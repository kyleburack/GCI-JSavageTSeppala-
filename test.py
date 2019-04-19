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
from keras.models import load_model

print("Loading model...")
model = load_model('trained_models/xception_model_1')

# image loading + preproccessing
datagen = ImageDataGenerator() ###### EDIT THIS LATER IF OVERFITTING BECOMES AN ISSUE

# certain hyperparameters
batch = 5

print("Loading test data...")
test_generator = datagen.flow_from_directory(
    directory= 'gci_data/test',
    target_size=(1024, 1024),
    color_mode="rgb",
    batch_size=batch,
    class_mode='categorical',
    shuffle=True,)

print("Compiling model...")
model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics =['accuracy'])

print("Evaluating model...")
scores = model.evaluate_generator(test_generator,verbose=1)
print(scores)
