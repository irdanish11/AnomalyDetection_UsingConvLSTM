# -*- coding: utf-8 -*-
"""
Created on Sat May 16 01:32:43 2020

@author: Danish

Dependencies: pip install natsort
              Sort list, reference: https://stackoverflow.com/questions/33159106/sort-filenames-in-directory-in-ascending-order
"""

import glob
import natsort
from preprocessing import ToJson, GlobalNormalization, rgb2Gray
import cv2
import numpy as np
import time
from tqdm import tqdm

def load_dataset_frames(frame_dirs, frames_ext='.jpg', save=True, name='FramesList', val_split=None, shuffle=False, time_dim=None):
    if shuffle:
        if type(time_dim)!=int:
            raise  TypeError('Invalid value given to `time_dim`, provide an integer e.g 16 i.e the total number of frames that would be stacked across time dimension')
    directories = []
    for dirc in frame_dirs:
         sub_dir = glob.glob(dirc+'/*')
         directories.extend(sub_dir)
    print('Reading Frame names from directories!')
    frames_lst = []
    for dirc in tqdm(directories):
        frames = glob.glob(dirc+'/*{0}'.format(frames_ext))
        frames = natsort.natsorted(frames,reverse=False)
        frames_lst.extend(frames)
    #Shuffle the data
    if shuffle:
        arr = np.array(np.array_split(frames_lst, len(frames_lst)//time_dim))
        np.random.shuffle(arr)
        flat_arr = arr.flatten()
        frames_lst = flat_arr.tolist()
    if save:
        ToJson(frames_lst, name, path='./', json_dir=False)
    if val_split:
        split = int(len(frames_lst)-len(frames_lst)*val_split)
        train_frames = frames_lst[0:split]
        val_frames = frames_lst[split:-1]
        return train_frames, val_frames
    else:
        return frames_lst
    
class BatchGenerator:
    def __init__(self, batch_size=None, frames_lst=None, batch_shape=None):
        self.batch_size = batch_size
        self.frames_lst = frames_lst
        self.batch_shape = batch_shape
        self.counter = 0;
    def get_nextBatch(self):
        frames_batch = self.frames_lst[self.counter:self.counter + self.batch_size]
        frames = []
        for f in frames_batch:
            gray = rgb2Gray(cv2.imread(f))
            frames.append(gray)
        frames = np.array(frames)
        #Aplying Global normalization to the batch
        frames = GlobalNormalization(frames)
        frames = np.reshape(frames, self.batch_shape)
        #incrementing the counter
        self.counter += self.batch_size
        return frames
    def frames2array(self, frame_names, time_dim=16, channel=1):
        frames = []
        for f in frame_names:
            gray = rgb2Gray(cv2.imread(f))
            frames.append(gray)
        frames = np.array(frames)
        #Aplying Global normalization to the batch
        frames = GlobalNormalization(frames)
        shape = np.shape(frames)
        samples = int(shape[2]/time_dim)
        #Slice the array from third axis, as the shape of array changes after GlobalNormalization to: (width,height,batch_size)
        frames = frames[:, :, 0:samples*16]
        frames = np.reshape(frames, (samples, shape[0], shape[1], time_dim, channel))
        return frames
        
class Timer:
    def __init__(self):
        self.begin = 0
    def restart(self):
        self.begin = time.time()
    def start(self):
        self.begin = time.time()
    def get_time_hhmmss(self, rem_batches):
        end = time.time()
        time_taken = end - self.begin
        reamin_time = time_taken*rem_batches
        #print('reamin time: '+str(reamin_time)+' Reamin Batches: '+str(rem_batches)+' Time Taken: '+str(time_taken))
        m, s = divmod(reamin_time, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        return time_str, time_taken