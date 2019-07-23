# import pandas as pd
import numpy as np
from scipy import signal
import scipy.io.wavfile as wav
import os
import time
import sys
import matplotlib.pyplot as plt
# import skimage.io
import platform
from keras.models import Sequential, load_model

#############################################
# add the file to test here #
filepath="/home/renyuming/voice_recognition/ALL_hd_random（10）/test/02 metalnode/190103_1309.wav"

def data_preprocess(filepath):
    (rate, data) = wav.read(filepath)
    data = np.transpose(data)[0]
    data = data[-132400:-1].copy()
    sample_frequency, segment_time, spectrogram = signal.spectrogram(data)
    spectrogram = np.asarray(spectrogram)
    spectrogram=np.expand_dims(spectrogram,axis=0)
    return spectrogram


test_x=data_preprocess(filepath)

model = load_model('voice_recog_spectrogram_preprcsess_10lei_1d_300epochs_01.h5')
result = model.predict_on_batch(test_x)
possibiliy=np.max(result)
class_number = np.argmax(result)
list_of_class={0:"diode",1:"metalnode",2:"qiangkaiguan",3:"mouse",4:"dianluban",5:"libattery",
6:"charger",7:"A-wav",8:"qiangchazuo",9:"netport"}

print("The object is",list_of_class[class_number],", possibiliy:",possibiliy*100,"%.")