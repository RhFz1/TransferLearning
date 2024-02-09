#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:54:51 2024

@author: syed
"""
import os
from model import return_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from data import IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, INPUT_SHAPE, N_CLASSES, BATCH_SIZE
from data import traingen, validgen

from model import model_path


def train_model():
    
    optim_1 = Adam(learning_rate=0.001)
    
    n_steps = traingen.samples // BATCH_SIZE
    n_val_steps = validgen.samples // BATCH_SIZE
    n_epochs = 50
    
    # First we'll train the model without Fine-tuning
    vgg_model = return_model(INPUT_SHAPE, N_CLASSES, optim_1, fine_tune=0)
    
    # ModelCheckpoint callback - save best weights
    tl_checkpoint_1 = ModelCheckpoint(filepath=model_path,
                                      save_best_only=True,
                                      verbose=1)
    
    # EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=10,
                               restore_best_weights=True,
                               mode='min')
    
    vgg_history = vgg_model.fit(traingen,
                                batch_size=BATCH_SIZE,
                                epochs=n_epochs,
                                validation_data=validgen,
                                steps_per_epoch=n_steps,
                                validation_steps=n_val_steps,
                                callbacks=[tl_checkpoint_1, early_stop],
                                verbose=1)
    
