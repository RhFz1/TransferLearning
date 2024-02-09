#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:52:10 2024

@author: syed
"""


from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
N_CLASSES = 2
BATCH_SIZE = 4

data_dir = '/home/syed/Desktop/MyCode/images' # input path to the data.


train_generator = ImageDataGenerator(rotation_range=90, 
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.5, 
                                     height_shift_range=0.5,
                                     horizontal_flip=True, 
                                     vertical_flip=True,
                                     validation_split=0.15) # VGG16 preprocessing

#test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data_dir = data_dir
test_data_dir = data_dir

traingen = train_generator.flow_from_directory(train_data_dir,
                                               target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                               class_mode='categorical',
                                               subset='training',
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               seed=42)

validgen = train_generator.flow_from_directory(train_data_dir,
                                               target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
                                               class_mode='categorical',
                                               subset='validation',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               seed=42)









