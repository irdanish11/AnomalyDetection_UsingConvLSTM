# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:12:22 2020

@author: Danish
"""

import numpy as np
import glob
import os 
import math
import cv2
from tqdm import tqdm
import json


def Frame_Extractor(v_file, path='./', ext='.avi', frames_dir='train_1', extract_rate='all', frames_ext='.jpg', resize=False, size=None):
    """
    A method which extracts the frames from the guven video. It can ex

    Parameters
    ----------
    
    v_file : str
        Name of the video file, without extension.
    
    path : str
        Path to the video file, if the video is in the current working directory do not specify this argument.
    
    ext : str, optional
        Extension of the given Video File e.g `.avi`, `.mp4`. The default is '.avi'.
    
    frames_dir : str, optional
        Path to the directory where frames will be saved. The default is 'train_1'.
    
    extract_rate : int or str, optional
        This argument specifies how many frames should be extrcated from each 1 second of video. If the value is 
        `all` it will etract all the frames in every second i.e if the frame rate of video is 25 fps it will
        extrcat all 25 frames. Other wise specify a number if you want to etract specific numbers of frames
        per each second e.g if 5 is given it will extrcat 5 frames from each 1 second. The default is `all`.
    
    frames_ext : str, optional
        The extension for the extracted frames/images e.h '.tif' or '.jpg'. The default is '.jpg'.

    Returns
    -------
    None.

    """
    if resize:
        if size==None:
            raise TypeError('size cannot be None when resize set to True. Provide a tuple of two ints (width, height)')
    os.makedirs(frames_dir, exist_ok=True)
    # capturing the video from the given path
    if ext not in v_file:
        v_file += ext
    cap = cv2.VideoCapture(path+v_file)
    
    frameRate = cap.get(5) #frame rate
    f_counter=0
    f_list = []
    if '.' in v_file:
        v_name = v_file.split('.')[0]
    else:
        v_name = vfile

    #duration = int(cap.get(7)/frameRate)
    os.makedirs(frames_dir+'/'+v_name, exist_ok=True)
    count = 0
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        frame = cv2.resize(frame, size)
        #incrementing frame number total
        f_counter += 1
        if type(extract_rate)==int:
            if extract_rate>frameRate:
                print('Frame rate of Given Video: {0} fps'.format(frameRate))
                raise ValueError('The value of `extract_rate` argument can not be greater than the Frame Rate of the video.')
            #Extract the frames according to given extract rate e.g if extract_rate=16, then 16 frames from each second.
            if extract_rate>1:
                f_list.append(frame)
                if f_counter%frameRate==0:
                    indicies = np.arange(0,frameRate)
                    np.random.shuffle(indicies)
                    indicies = np.sort(indicies[0:extract_rate]) #'{0}frame_'.format(int(count)) + v_name
                    for i in indicies:
                        filename = frames_dir + '/' + v_name + '/'+  v_name + '_frame{0}'.format(int(count)) + frames_ext;count+=1
                        cv2.imwrite(filename, f_list[int(i)])
                    f_list = []
                    #print(filename)
                else:
                    f_list.append(frame)
                # storing the frames in a new folder named train_1
                # filename = frames_dir + '/' + v_file+ '/'+"_frame{0}".format(count)+frames_ext;count+=1
                # cv2.imwrite(filename, frame)
            elif extract_rate==1:
                if (frameId % math.floor(frameRate) == 0):
                    filename = frames_dir + '/' + v_name + '/'+  v_name + '_frame{0}'.format(int(count)) + frames_ext;count+=1
                    cv2.imwrite(filename, frame)
        elif type(extract_rate)==str:
            if extract_rate=='all':
                # storing the frames in a new folder named train_1
                filename = frames_dir + '/' + v_name + '/'+  v_name + '_frame{0}'.format(int(count)) + frames_ext;count+=1
                cv2.imwrite(filename, frame)
            else:
                raise ValueError('Invalid Value for argument `extract_rate`, it can be either `all` or an integer value.')
    cap.release()    

def ReadFileNames(path, frames_ext='.tif'):
    """
    This method will retrieve the Folder/File names from the dataset.

    Parameters
    ----------
    path : string
      Location of the data set that needs to be preprocessed.

    Returns
    -------
    onlyfiles : list
      A list containing all the subfolder names in the dataset.
    file_names : list
      A list containing all the names of the frames/images from 'onlyfiles'.
    directories: list
      A list containing all the names of the folders containing the images.
    """
    directories = [name for name in os.listdir(path) if os.path.isdir(path+'/'+name)]
    onlyfiles = []
    file_names = []
    
    for i in range (len(directories)):
        files = glob.glob(path+'/'+directories[i]+'/*{0}'.format(frames_ext))
        names = []
        for file in files:
            names.append(file.split("\\")[1])
        file_names.append(names)
        onlyfiles.append(files)
    return onlyfiles, file_names, directories

def ToJson(obj, name, path='./', json_dir=False):
    """ Write the Given Object to JSON file on the given path and name.
        
        Arguments:
        ----------
        obj: The object/variable you want to write to JSON file, it can be list or dictionary.
        
        name: Name for the JSON file.
        
        path: Path in which you want to save the JSON file. If want to save the JSON file in the current
        directory, do not specify the value of this argument.
              
        json_dir: Boolean value if set to `True` a new directory with the name of `JSON` will be added at
        the end of the given path.
                  
        Returns:
        --------
        None"""
    if json_dir:
        os.makedirs(path+'/JSON', exist_ok=True)
        with open(path+'/JSON/{0}.json'.format(name), 'w') as f:
            json.dump(obj, f)
        f.close()
    elif not json_dir:
        if '.json' in name:
            pass
        else:
            name=name+'.json'
        with open(path+'/'+name, 'w') as f:
            json.dump(obj, f)
        f.close()

def ProcessImg(img_name, read_path, write=True, write_path=None, res_shape=(128,128)):
    if write and write_path is None:
        raise TypeError('The value of argument cannot be `None` when, `write` is set to True. Provide a valid path, where processed image should be stored!')
    img=cv2.imread(read_path)
    #img=img_to_array(img)
    #Resize the Image to (227,227)
    img=cv2.resize(img,res_shape)
    
    rgb_weights = [0.2989, 0.5870, 0.1140]
    gray = np.dot(img, rgb_weights)
    
    if write:
        os.makedirs(write_path, exist_ok=True)
        cv2.imwrite(write_path+'/'+img_name, gray) 
    return gray

def GlobalNormalization(img_list, name=None, path='Train_Data', save_data=True):
    img_arr = np.array(img_list)
    batch,height,width = img_arr.shape
    #Reshape to (227,227,batch_size)
    img_arr.resize(height,width,batch)
    #Normalize
    img_arr=(img_arr-img_arr.mean())/(img_arr.std())
    #Clip negative Values
    img_arr=np.clip(img_arr,0,1)
    if save_data:
        if name==None:
            raise TypeError('The value of the `name` argument cannot be `None` type, when `save_data` is set to True. Provide value with `str` datatype.')
        if '.npy' not in name:
            name += '.npy'
        os.makedirs(path, exist_ok=True)
        np.save(path+'/'+name, img_arr)
        print('\n------ Data Save Succefully at this path: {0} ------\n'.format(path))
    return img_arr

def Vid2Frame(vid_path, frames_dir, ext_vid='.mp4', frames_ext='.jpg', extract_rate='all'):
    vids = glob.glob(vid_path+'/*{0}'.format(ext_vid))
    
    for vid in tqdm(vids):
        path = os.path.dirname(vid)+'/'
        v_file = os.path.basename(vid)
        Frame_Extractor(v_file, path=path, ext=ext_vid, frames_dir=frames_dir, extract_rate=extract_rate, 
                        frames_ext=frames_ext, resize=True, size=(240,360))  
        
def Fit_Preprocessing(path, frames_ext):
    if frames_ext is None:
        raise TypeError('Invalid Value for argument `frames_ext`, it cannot be None. Give proper extensions of the frames e.g: `.tif` or `.png` etc!')
    print('\n\nProcessing Images in this Dataset Path: {0}\n'.format(path))
    onlyfiles, file_names, dirs = ReadFileNames(path, frames_ext)
    img_list = []
    for i in tqdm(range(len(onlyfiles))):
        images = onlyfiles[i]
        count = 0
        for img in images:
            img.split('/')
            img_name = dirs[i]+'_'+file_names[i][count]
            write_path = 'ProcessedImages/'+path.split('/')[1]
            gray = ProcessImg(img_name, read_path=img, write=True, 
                              write_path=write_path, res_shape=(227,227))
            img_list.append(gray)
            count += 1    
    return img_list
    
if __name__=='__main__':    
    vid_paths = ['./Dataset/chute02']
    for vid_path in vid_paths:
        print('\n\nExtracting Frame from the videos, Path: {0}\n'.format(vid_path))
        frames_dir = 'Frames/'+vid_path
        Vid2Frame(vid_path, frames_dir, ext_vid='.avi', frames_ext='.jpg', 
                  extract_rate=16)
    print('\n-------- Frames are Extracted from All Videos Succesfully! --------\n')  
    


    # paths = ['path_to_frames/Train', 'path_to_frames/Test']
    # for path in paths:
    #     img_list = Fit_Preprocessing(path, frames_ext='.jpg')
    #     name='Test_{0}.npy'.format(path.split('/')[1])
    #     img_arr = GlobalNormalization(img_list, name, path='Test_Data', save_data=True)