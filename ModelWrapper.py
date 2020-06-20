# -*- coding: utf-8 -*-
"""
Created on Sun May  10 18:14:28 2020

@author: Danish
"""

from tensorflow.keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose, Input
from tensorflow.keras.models import Model
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
import sys
from BatchGenerator import BatchGenerator, Timer
from preprocessing import ToPickle, ToJson
import os
from tqdm import tqdm
from tensorflow.keras.models import load_model

def BuildModel(input_shape=(227,227,10,1)):
    if len(input_shape) != 4 or type(input_shape) != tuple:
        raise ValueError('Invalid value given to the argument `input_shape`, it must be a `tuple` containing 4 values in this manner: (height, width, frames_per_input, channels)')
    input = Input(shape=input_shape)
    #Spatial Encoder
    spatial_enc = Conv3D(filters=128, kernel_size=(11,11,1), strides=(4,4,1), padding='valid', activation='tanh')(input)
    spatial_enc = Conv3D(filters=64, kernel_size=(5,5,1), strides=(2,2,1), padding='valid', activation='tanh')(spatial_enc)

    #Temporal Encoder
    temporal_enc = ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True)(spatial_enc)
    temporal_enc = ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True)(temporal_enc)

    #Temporal Decoder
    temporal_dec = ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5)(temporal_enc)

    #Spatial Decoder
    spatial_dec = Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh')(temporal_dec)
    spatial_dec = Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh')(spatial_dec)

    # Model
    model = Model(inputs=input, outputs=spatial_dec)
    encoder = Model(inputs=input, outputs=temporal_enc)
    return model, encoder

def load_model_weights(path, model_path, encoder_path):
    model = load_model(path+'/'+model_path)
    encoder = load_model(path+'/'+encoder_path)
    return [model, encoder]

def TF_GPUsetup(GB=4):
    """
    Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU. Often Needed
    When GPU run out of memory. It would be one of the solution for the issue: Failed to 
    get convolution algorithm. This is probably because cuDNN failed to initialize,

    Parameters
    ----------
    GB : int, optional
        The amount of GPU memory you want to use. It is recommended to use  1 GB
        less than your total GPU memory. The default is 4.

    Returns
    -------
    None.

    """
    if type(GB)!=int:
        raise TypeError('Type of Parameter `GB` must be `int` and it should be 1 GB less than your GPU memory')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    config = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*GB))]
    if gpus:
      # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
      try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], config)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
    print('\nTensorflow GPU installed: '+str(tf.test.is_built_with_cuda()))
    print('Is Tensorflow using GPU: '+str(tf.test.is_gpu_available()))
    
def PrepareData(X_train, re_shape=(-1,227,227,10)):
    frames = X_train.shape[2]
    #Need to make number of frames divisible by 10
    frames -= frames%10
    X_train=X_train[:,:,:frames]
    X_train=X_train.reshape(re_shape)
    X_train=np.expand_dims(X_train,axis=4)
    return X_train
    

def GetTrainData(name, re_shape=(-1,227,227,10)):
    if type(name)!=str:
        raise TypeError('Provide a valid name of `string` datatype, to the `.npy` file.')
    if '.npy' not in name:
        name += '.npy'
    X_train = np.load(name)
    return PrepareData(X_train, re_shape)

def PrintInline(string):
    sys.stdout.write('\r'+string)
    sys.stdout.flush() 

def ImgProcess(frame, shape=(227,227)):
    frame=cv2.resize(frame,shape)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    gray = np.dot(frame, rgb_weights)
    gray=(gray-gray.mean())/gray.std()
    gray=np.clip(gray,0,1)
    return gray


def MSE(x1,x2):
    """
    Compute Euclidean Distance Loss between input frame and the reconstructed frame and then
    compute the Mean Squared Error

    Parameters
    ----------
    x1 : TYPE
        DESCRIPTION.
    x2 : TYPE
        DESCRIPTION.

    Returns
    -------
    mean_dist : TYPE
        DESCRIPTION.

    """
    diff=x1-x2
    a,b,c,d,e=diff.shape
    n_samples=a*b*c*d*e
    sq_diff=diff**2
    Sum=sq_diff.sum()
    dist=np.sqrt(Sum)
    mean_dist=dist/n_samples
    
    return mean_dist


def _callback(epoch_loss, global_loss, ckpt_path, model, encoder, e_stop, es_log, patience, min_delta):
    #Implementing early stopping
    if e_stop:
        d = global_loss-epoch_loss
        #print(d, global_loss,epoch_loss)
        if d<min_delta:
            es_log += 1
        else:
            path = os.path.dirname(ckpt_path)
            os.makedirs(path, exist_ok=True)
            model.save(ckpt_path)
            encoder.save(path+'/encoder.h5')
            print('Loss imporoved, Model saved to checkpoints file!')
            es_log=0
            global_loss = epoch_loss
        if es_log>patience:
            print('Training Terminated as loss was not improving, Early Stopping count: {0}'.format(patience))
            return global_loss, es_log, True
        else:
            return global_loss, es_log, False
    else:
        #Saving checkpoints
        if epoch_loss<global_loss:
            path = os.path.dirname(ckpt_path)
            os.makedirs(path, exist_ok=True)
            model.save(ckpt_path)
            encoder.save(path+'/encoder.h5')
            print('Loss imporoved, Model saved to checkpoints file!')

def save_processed_batches(train_frames, batch_size=192, batch_shape=(12, 315, 235, 16, 1), path='./npy_data'):
    """
    Preproces the images and save them in compressed numpy format, so that images don't need to be
    processed at every epoch thus resulting in reducing the training time, but this methid does requires
    a lot of storage'
    Parameters
    ----------
    train_frames : TYPE
        DESCRIPTION.
    val_frames : TYPE
        DESCRIPTION.
    batch_size : TYPE, optional
        DESCRIPTION. The default is 192.
    batch_shape : TYPE, optional
        DESCRIPTION. The default is (12, 315, 235, 16, 1).
    path : TYPE, optional
        DESCRIPTION. The default is './npy_data'.

    Returns
    -------
    None.

    """
    bg = BatchGenerator(batch_size, train_frames, batch_shape)
    total_batches = int(len(train_frames)/batch_size)
    os.makedirs(path+'/'+'train_batches', exist_ok=True)
    train_batches = {}
    train_lst = []
    for i in tqdm(range(1, total_batches+1)): 
        train_batches['Batch'.format(i)] = [(i-1)*batch_size, i*batch_size]
        X_train = bg.get_nextBatch(i)
        X_train = np.array(X_train, dtype=np.float16)
        name = path+'/'+'train_batches/'+'Batch{0}.npz'.format(i)
        np.savez_compressed(name, X_train)
        train_lst.append(name)
    ToPickle(train_batches, name='train_batches_info.pickle')
    ToJson(train_lst, name='train_lst.json')
    return train_lst


def fit(model_lst, train_frames, epochs, ckpt_path, dp_type='pre', val_frames=None, batch_size=192, batch_shape=(12, 315, 235, 16, 1), e_stop=False, patience=3, min_delta=0.0, init_epoch=1, batch_n=1):
    """
    Parameters
    ----------
    model_lst : TYPE
        DESCRIPTION.
    train_frames : TYPE
        DESCRIPTION.
    epochs : TYPE
        DESCRIPTION.
    ckpt_path : TYPE
        DESCRIPTION.
    dp_type : string type
        Specifies the data preprocessing to be used, either `pre` or `post`. If value is `pre`, then you
        have to provide a list of paths to the preprocessed numpy arrays to argument `train_frames` which 
        are saved to the disk using save_processed_batches() method this will save train time, If you want 
        to preprocess images on every epoch then provide `post` to the dp_type and provide a list of paths
        to each and every image to argument `train_frames`.
    val_frames : TYPE, optional
        DESCRIPTION. The default is None.
    batch_size : TYPE, optional
        DESCRIPTION. The default is 192.
    batch_shape : TYPE, optional
        DESCRIPTION. The default is (12, 315, 235, 16, 1).
    e_stop : TYPE, optional
        DESCRIPTION. The default is False.
    patience : TYPE, optional
        DESCRIPTION. The default is 3.
    min_delta : TYPE, optional
        DESCRIPTION. The default is 0.0.
    init_epoch : TYPE, optional
        DESCRIPTION. Use to resume the training. In case of resuming you have to provide the model that
        was saved by the previous epoch. The number should be equal to the number of times previous model
        that you are giving now has been trained. The default is 1.
    batch_n : TYPE, optional
        DESCRIPTION. The number of batches your previous model was trained for the last epoch, if you are
        not sure just don't set this parameter or just provide 1. The default is 1.

    Raises
    ------
    TypeError

    Returns
    -------
    None.

    """
    model=model_lst[0]; encoder=model_lst[1]
    if e_stop:
        if type(patience)!=int:
            raise TypeError('Inavlid value given to es_count, provide an integer value.')     
    #TF_GPUsetup(5)
    time = Timer()
    global_loss = 10e5
    es_log=0
    print('Training Will Start Now')
    for epoch in range(init_epoch, epochs+1):
        print('\n')
        epoch_acc = 0
        epoch_loss = 0
        total_time = 0
        if dp_type=='post':
            bg = BatchGenerator(batch_size, train_frames, batch_shape)
            total_batches = int(len(train_frames)/batch_size)
        elif dp_type=='pre':
            total_batches = len(train_frames)
        epoch_data = list(range(0,50))
        if epoch==init_epoch:
            resume = True
            batch=batch_n
            print('Previous model found, training will resume now from: Epoch: {0}, and Batch: {1}'.format(epoch, batch))
        else:
            batch=1
        for i in range(batch, total_batches): 
            batch_data = {'Batch':[],'Loss':[], 'Accuracy':[]}
            #print('Gett')
            if dp_type=='post':
                X_train = bg.get_nextBatch(i)
            elif dp_type=='pre':
                npzfile = np.load(train_frames[i])
                X_train = npzfile[npzfile.files[0]]
            #start timer
            time.start()
            """Training Model"""
            metrics = model.train_on_batch(X_train, X_train)
            #end timer on batch and calculate estimated time on remaining batches
            time_remain, time_taken = time.get_time_hhmmss(total_batches-i)
            str1 = 'Epoch {0}/{1}, Batch {2}/{3}, Time Taken By 1 Batch: {4:.2} sec. '.format(epoch, epochs, i, total_batches, time_taken)
            if type(metrics)==list:
                batch_loss = metrics[0]; batch_acc = metrics[1]
                epoch_loss += batch_loss
                epoch_acc += batch_acc
                str2 = 'Est Time Remaining: {0}, Loss: {1:.5}, Accuracy: {2:.3}'.format(time_remain, batch_loss, batch_acc)
            else:
                batch_loss = metrics
                str2 = 'Est Time Remaining: {0}, Loss: {1:.5}'.format(time_remain, batch_loss)
            PrintInline(str1+str2)
            total_time += time_taken
            batch_data['Batch'].append(i); batch_data['Loss'].append(batch_loss); batch_data['Accuracy'].append(batch_acc);  
            epoch_data[epoch] = batch_data
            ToPickle(epoch_data, 'Batch_Data')
            if i%2000==0:
                path = os.path.dirname(ckpt_path)
                f_name = 'Batch_'+os.path.basename(ckpt_path)
                os.makedirs(path+'/'+f_name, exist_ok=True)
                model.save(ckpt_path)
                encoder.save(path+'/batch_encoder.h5')
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        if type(metrics)==list:
            print('\nLoss on epoch: {0:.5}, Accuracy on Epoch: {1:.3} Total time taken by epoch: {2}'.format(epoch_loss/total_batches, epoch_acc/total_batches, time_str))
        else:
            print('\nLoss on epoch: {0:.5}, Total time taken by epoch: {1}'.format(epoch_loss/total_batches, time_str))
        
        if val_frames:
            if epoch==1 or resume:
                bg = BatchGenerator(batch_size, train_frames, batch_shape)
                print('Reading Validation Data')
                val_data = bg.frames2array(val_frames, time_dim=batch_shape[3], channel=batch_shape[4])
            print('Performing Validation')
            val_metrics = model.evaluate(val_data, val_data, batch_size=4)
            if type(val_metrics)==list:
                val_loss = val_metrics[0]; val_acc = val_metrics[1]
                print('Validation Loss: {0:.5}, Validation Accuracy: {1:.3}, Number of Validation Samples: {2}'.format(val_loss, val_acc, len(val_data)))
            else:
                val_loss = val_metrics
                print('Validation Loss: {0:.5}'.format(val_loss))
            global_loss, es_log, br = _callback(val_loss, global_loss, ckpt_path, model, encoder, e_stop, es_log, patience, min_delta)
        else:
            global_loss, es_log, br = _callback(epoch_loss, global_loss, ckpt_path, model, encoder, e_stop, es_log, patience, min_delta)
        history = {'Train_loss':[], 'Train_acc': [], 'val_loss':[], 'val_acc':[], 'Epoch':[]}
        history['Train_loss'].append(epoch_loss); history['Train_acc'].append(epoch_acc); history['Epoch'].append(epoch);
        history['Train_loss'].append(val_loss); history['Train_loss'].append(val_acc);
        ToPickle(history, name='history')
        if br:
            break
            
