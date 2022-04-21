import matplotlib
matplotlib.use('agg')


import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as io_wav
from detect_peaks import detect_peaks
import os
import os.path
import gc
import re
import tgt
import csv
import datetime
import scipy
import pickle
import skimage
import cv2
import random
random.seed(17)


from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from tensorflow.python.framework import device_spec as tf_device
from keras.callbacks import ReduceLROnPlateau
import wgfunctions
    
# additional requirement: SPTK 3.8 or above in PATH

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ResNet model
from residualnetworks import ResNet50_regression

# do not use all GPU memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True 
set_session(tf.compat.v1.Session(config=config))


# read_ult reads in *.ult file from AAA
def read_ult(filename, NumVectors, PixPerVector):
    # read binary file
    ult_data = np.fromfile(filename, dtype='uint8')
    ult_data = np.reshape(ult_data, (-1, NumVectors, PixPerVector))
    return ult_data


# read_psync_and_correct_ult reads *_sync.wav and finds the rising edge of the pulses
# if there was a '3 pulses bug' during the recording,
# it removes the first three frames from the ultrasound data
def read_psync_and_correct_ult(filename, ult_data):
    (Fs, sync_data_orig) = io_wav.read(filename)
    sync_data = sync_data_orig.copy()

    # clip
    sync_threshold = np.max(sync_data) * 0.6
    for s in range(len(sync_data)):
        if sync_data[s] > sync_threshold:
            sync_data[s] = sync_threshold

    # find peeks
    peakind1 = detect_peaks(sync_data, mph=0.9*sync_threshold, mpd=10, threshold=0, edge='rising')
    
    '''
    # figure for debugging
    plt.figure(figsize=(18,4))
    plt.plot(sync_data)
    plt.plot(np.gradient(sync_data), 'r')
    for i in range(len(peakind1)):
        plt.plot(peakind1[i], sync_data[peakind1[i]], 'gx')
        # plt.plot(peakind2[i], sync_data[peakind2[i]], 'r*')
    plt.xlim(2000, 6000)
    plt.show()    
    '''
    
    # this is a know bug: there are three pulses, after which there is a 2-300 ms silence, 
    # and the pulses continue again
    if (np.abs( (peakind1[3] - peakind1[2]) - (peakind1[2] - peakind1[1]) ) / Fs) > 0.2:
        bug_log = 'first 3 pulses omitted from sync and ultrasound data: ' + \
            str(peakind1[0] / Fs) + 's, ' + str(peakind1[1] / Fs) + 's, ' + str(peakind1[2] / Fs) + 's'
        print(bug_log)
        
        peakind1 = peakind1[3:]
        ult_data = ult_data[3:]
    
    for i in range(1, len(peakind1) - 2):
        # if there is a significant difference between peak distances, raise error
        if np.abs( (peakind1[i + 2] - peakind1[i + 1]) - (peakind1[i + 1] - peakind1[i]) ) > 1:
            bug_log = 'pulse locations: ' + str(peakind1[i]) + ', ' + str(peakind1[i + 1]) + ', ' +  str(peakind1[i + 2])
            print(bug_log)
            bug_log = 'distances: ' + str(peakind1[i + 1] - peakind1[i]) + ', ' + str(peakind1[i + 2] - peakind1[i + 1])
            print(bug_log)
            
            raise ValueError('pulse sync data contains wrong pulses, check it manually!')
    
    return ([p for p in peakind1], ult_data)


# read_meta reads in *.txt ult metadata file from AAA
def read_param(filename):    
    NumVectors = 0
    PixPerVector = 0
    # read metadata from txt
    for line in open(filename):
        # 1st line: NumVectors=64
        if "NumVectors" in line:
            NumVectors = int(line[11:])
        # 2nd line: PixPerVector=842
        if "PixPerVector" in line:
            PixPerVector = int(line[13:])
        # 3rd line: ZeroOffset=210
        if "ZeroOffset" in line:
            ZeroOffset = int(line[11:])
        # 5th line: Angle=0,025
        if "Angle" in line:
            Angle = float(line[6:].replace(',', '.'))
        # 8th line: FramesPerSec=82,926
        # Warning: this FramesPerSec value is usually not real, use calculate_FramesPerSec function instead!
        if "FramesPerSec" in line:
            FramesPerSec = float(line[13:].replace(',', '.'))
        # 9th line: first frame
        # TimeInSecsOfFirstFrame=0.95846
        if "TimeInSecsOfFirstFrame" in line:
            TimeInSecsOfFirstFrame = float(line[23:].replace(',', '.'))
    
    return (NumVectors, PixPerVector, ZeroOffset, Angle, FramesPerSec, TimeInSecsOfFirstFrame)

def cut_and_resample_wav(filename_wav_in, Fs_target):
    filename_no_ext = filename_wav_in.replace('.wav', '')
    
    filename_param = filename_no_ext + '.param'
    filename_wav_out = filename_no_ext + '_cut_22k.wav'
    
    # resample speech using SoX
    command = 'sox ' + filename_wav_in + ' -r ' + str(Fs_target) + ' ' + \
              filename_no_ext + '_22k.wav'
    call(command, shell=True)
    
    # volume normalization using SoX
    command = 'sox --norm=-3 ' + filename_no_ext + '_22k.wav' + ' ' + \
              filename_no_ext + '_22k_volnorm.wav'
    call(command, shell=True)
    
    # cut from wav the signal the part where there are ultrasound frames
    (NumVectors, PixPerVector, ZeroOffset, Angle, FramesPerSec, TimeInSecsOfFirstFrame) = read_param(filename_param)
    (speech_wav_data, Fs_wav) = read_wav(filename_no_ext + '_22k_volnorm.wav')
    init_offset = int(TimeInSecsOfFirstFrame * Fs_wav) # initial offset in samples
    speech_wav_data = speech_wav_data[init_offset - hop_length_UTI : ]
    write_wav(speech_wav_data, Fs_wav, filename_wav_out)
    
    # remove temp files
    os.remove(filename_no_ext + '_22k.wav')
    os.remove(filename_no_ext + '_22k_volnorm.wav')
    
    print(filename_no_ext + ' - resampled, volume normalized, and cut to start with ultrasound')
    

# WaveGlow / Tacotron2 / STFT parameters
samplingFrequency = 22050
n_melspec = 80
hop_length_UTI = 270 # 12 ms, corresponding to 81.5 fps at 22050 Hz sampling
stft = wgfunctions.TacotronSTFT(filter_length=1024, hop_length=hop_length_UTI, \
    win_length=1024, n_mel_channels=n_melspec, sampling_rate=samplingFrequency, \
    mel_fmin=0, mel_fmax=8000)


# parameters of ultrasound images, from .param file
framesPerSec = 81.5
n_lines = 64
n_pixels = 842

# reduce ultrasound image resolution
n_pixels_reduced = 128



# TODO: modify this according to your data path
dir_base = '/shared/UltraSuite_TaL/TaL80/core/'
##### training data
# - females: 01fi, 02fe, 09fe
# - males: 03mn, 04me, 05ms, 06fe, 07me, 08me, 10me
# speakers = ['01fi', '02fe', '03mn', '04me', '05ms', '06fe', '07me', '08me', '09fe', '10me']
speakers = ['01fi']


for speaker in speakers:
    
    # collect all possible ult files
    ult_files_all = []
    dir_data = dir_base + speaker + '/'
    if os.path.isdir(dir_data):
        for file in sorted(os.listdir(dir_data)):
            # collect _aud and _xaud files
            if file.endswith('aud.ult'):
                ult_files_all += [dir_data + file[:-4]]
    
    # randomize the order of files
    random.shuffle(ult_files_all)
    
    # temp: only first 10 sentence
    ult_files_all = ult_files_all[0:10]
    
    ult_files = dict()
    ult = dict()
    melspec = dict()
    ultmel_size = dict()
    
    # train: first 80% of sentences
    ult_files['train'] = ult_files_all[0:int(0.8*len(ult_files_all))]
    # valid: next 10% of sentences
    ult_files['valid'] = ult_files_all[int(0.8*len(ult_files_all)):int(0.9*len(ult_files_all))]
    # valid: last 10% of sentences
    ult_files['test'] = ult_files_all[int(0.9*len(ult_files_all)):]
    
     #print('train files: ', ult_files['train'])
     #print('valid files: ', ult_files['valid'])
    
    for train_valid in ['train', 'valid']:
        n_max_ultrasound_frames = len(ult_files[train_valid]) * 500
        ult[train_valid] = np.empty((n_max_ultrasound_frames, n_lines, n_pixels_reduced))
        melspec[train_valid] = np.empty((n_max_ultrasound_frames, n_melspec))
        ultmel_size[train_valid] = 0
        
        # load all training/validation data
        for basefile in ult_files[train_valid]:
            try:
                ult_data = read_ult(basefile + '.ult', n_lines, n_pixels)
                
          
                
                # resample and cut if necessary
                if not os.path.isfile(basefile + '_cut_22k.wav'):
                    cut_and_resample_wav(basefile + '.wav', samplingFrequency)
                    
                # load using mel_sample
                mel_data = wgfunctions.get_mel(basefile + '_cut_22k.wav', stft)
                mel_data = np.fliplr(np.rot90(mel_data.data.numpy(), axes=(1, 0)))
                
            except ValueError as e:
                print("wrong psync data, check manually!", e)
            else:
                ultmel_len = np.min((len(ult_data),len(mel_data)))
                ult_data = ult_data[0:ultmel_len]
                mel_data = mel_data[0:ultmel_len]
                
                print(basefile, ult_data.shape, mel_data.shape)
                
                if ultmel_size[train_valid] + ultmel_len > n_max_ultrasound_frames:
                    print('data too large', n_max_ultrasound_frames, ultmel_size[train_valid], ultmel_len)
                    raise
                
                for i in range(ultmel_len):
                    ult[train_valid][ultmel_size[train_valid] + i] = skimage.transform.resize(ult_data[i], (n_lines, n_pixels_reduced), preserve_range=True) / 255
                    
                #plt.imshow (ult [train_valid][ultmel_size[train_valid]+3])
                #plt.gray()
                #plt.show()
                
                melspec[train_valid][ultmel_size[train_valid] : ultmel_size[train_valid] + ultmel_len] = mel_data
                ultmel_size[train_valid] += ultmel_len
                
                print('n_frames_all: ', ultmel_size[train_valid])


        ult[train_valid] = ult[train_valid][0 : ultmel_size[train_valid]]
        melspec[train_valid] = melspec[train_valid][0 : ultmel_size[train_valid]]

        # input: already scaled to [0,1] range
        # rescale to [-1,1]
        ult[train_valid] -= 0.5
        ult[train_valid] *= 2
        # reshape ult for CNN
        ult[train_valid] = np.reshape(ult[train_valid], (-1, n_lines, n_pixels_reduced, 1))
        
    print(ult['train'].shape)
    # target: normalization to zero mean, unit variance
    melspec_scaler = StandardScaler(with_mean=True, with_std=True)
    # melspec['train'] = melspec_scaler.fit_transform(melspec['train'].reshape(-1, 1)).ravel()
    melspec['train'] = melspec_scaler.fit_transform(melspec['train'])
    melspec['valid'] = melspec_scaler.transform(melspec['valid'])

        
    # get ResNet model
    model = ResNet50_regression((ult['train'].shape[1:]), n_melspec)

    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam')

    print(model.summary())
        

 # early stopping to avoid over-training
    # csv logger
    current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() )
    print(current_date)
    
   
    model_name = 'models/ResNet' + speaker + '_' + current_date
    
    # callbacks
    earlystopper = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=3, verbose=1, mode='auto')
    lrr = ReduceLROnPlateau(monitor='val_mean_squared_error', patience=2, verbose=1, factor=0.5, min_lr=0.0001) 
    logger = CSVLogger(model_name + '.csv', append=True, separator=';')
    checkp = ModelCheckpoint(model_name + '_weights_best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')


    # save model
    model_json = model.to_json()
    with open(model_name + '_model.json', "w") as json_file:
        json_file.write(model_json)

    # serialize scalers to pickle
    pickle.dump(melspec_scaler, open(model_name + '_melspec_scaler.sav', 'wb'))

    # Run training
    history = model.fit(ult['train'], melspec['train'],
                            epochs = 100, batch_size = 128, shuffle = True, verbose = 1,
                            validation_data=(ult['valid'], melspec['valid']),
                            callbacks = [earlystopper, lrr, logger, checkp])
        
        # here the training of ResNet is finished
