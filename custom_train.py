# -*- coding: utf-8 -*-
"""
Created on Sat May 16 04:27:50 2020

@author: Danish

Dependencies:   pip install natsort
                pip install tqdm
                pip install Pillow
                pip install opencv-python
"""

from BatchGenerator import load_dataset_frames
import ModelWrapper as mp
from tensorflow.keras.models import load_model

ckpt_path = './checkpoints/Autoencoder.h5'
model, encoder = mp.BuildModel(input_shape=(315, 235, 16, 1)) 
model_lst = [model, encoder]

frame_dirs = ['./Frames/Dataset/chute02']
train_frames, val_frames = load_dataset_frames(frame_dirs, frames_ext='.jpg', save=True, 
                                               name='FramesList', val_split=0.1)#0.004  
#Preprocessing data to save time
train_frames = mp.save_processed_batches(train_frames, batch_size=192, batch_shape=(12, 315, 235, 16, 1), 
                          path='./npy_data')

model.summary()
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
""" For resuming the model uncomment the following line & comment the model.compile 
    line, and specify the init_epoch argument to the number of epochs completed+1"""
#model = load_model(ckpt_path)
#mp.TF_GPUsetup()
mp.fit(model_lst, train_frames, epochs=100,  val_frames=val_frames, dp_type='pre', ckpt_path = ckpt_path,
       batch_size=192, batch_shape=(12, 315, 235, 16, 1), e_stop=True, patience=5, min_delta=0.00001,
       init_epoch=1, batch_n=1)  
 

# import tensorflow as tf

# assert tf.test.is_gpu_available()
# assert tf.test.is_built_with_cuda()


