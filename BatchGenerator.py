# -*- coding: utf-8 -*-
"""
Created on Sat May 16 01:32:43 2020

@author: Danish

Dependencies: pip install natsort
              Sort list, reference: https://stackoverflow.com/questions/33159106/sort-filenames-in-directory-in-ascending-order
"""

import glob
import natsort
from preprocessing import ToJson
import cv2
import numpy as np

def load_dataset_frames(frame_dirs, frames_ext='.jpg', save=True, name='FramesList'):
    directories = []
    for dirc in frame_dirs:
         sub_dir = glob.glob(dirc+'/*')
         directories.extend(sub_dir)
         
    frames_lst = []
    for dirc in directories:
        frames = glob.glob(dirc+'/*{0}'.format(frames_ext))
        frames = natsort.natsorted(frames,reverse=False)
        frames_lst.extend(frames)
    if save:
        ToJson(frames_lst, name, path='./', json_dir=False)
    return frames_lst
    
class BatchGenerator:
    def __init__(self, batch_size, frames_lst, batch_shape):
        self.batch_size = batch_size
        self.frames_lst = frames_lst
        self.batch_shape = batch_shape
        self.counter = 0;
    def get_nextBatch(self):
        frames_batch = self.frames_lst[self.counter:self.counter + self.batch_size]
        frames = []
        for f in frames_batch:
            frames.append(cv2.imread(f))
        frames = np.array(frames)
        frames = np.reshape(frames, self.batch_shape)
        self.counter += self.batch_size
        return frames