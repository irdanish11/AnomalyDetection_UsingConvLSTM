# -*- coding: utf-8 -*-
"""
Created on Sat May 16 04:27:50 2020

@author: Danish
"""

from BatchGenerator import BatchGenerator, load_dataset_frames

frame_dirs = ['./Frames/Dataset/chute02']
frames_lst = load_dataset_frames(frame_dirs, frames_ext='.jpg', save=True, name='FramesList')    
 
batch_size=256  
######### Following four lines of code should go inside the epoch loop ######     
''' Otherwise if you initialize BatchGenerator class out of the epoch it will give btaches
    for only 1 epoch, because at the last call of 1st epoch variable self.counter will reach 
    its upper bound, and on 2nd epoch we will not get any batches, so we have initialize the 
    class on every new epoch.''' 
bg = BatchGenerator(batch_size, frames_lst, batch_shape=(16, 16, 360, 240, 3))
batches = []
for i in range(int(len(frames_lst)/batch_size)):   
    batches.append(bg.get_nextBatch())