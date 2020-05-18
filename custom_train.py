# -*- coding: utf-8 -*-
"""
Created on Sat May 16 04:27:50 2020

@author: Danish
"""

from BatchGenerator import load_dataset_frames
import ModelWrapper as mp


model = mp.BuildModel(input_shape=(315, 235, 16, 1)) 


frame_dirs = ['./Frames/Dataset/chute02']
train_frames, val_frames = load_dataset_frames(frame_dirs, frames_ext='.jpg', save=True, 
                                               name='FramesList', val_split=0.004)  

model.summary()
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
mp.fit(model, train_frames, epochs=100, ckpt_path='./checkpoints/Autoencoder.h5', val_frames=val_frames, 
       batch_size=192, batch_shape=(12, 315, 235, 16, 1), e_stop=True, patience=5, min_delta=0.00001)  
 


    