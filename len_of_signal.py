import numpy as np
from scipy import signal
import scipy.io.wavfile as wav
import os
import time
import sys
from keras.utils.np_utils import to_categorical
#import matplotlib.pyplot as plt
#import skimage.io

ROOT_DIR=os.path.abspath('.')
wav_path = os.path.join(ROOT_DIR,"hd_signal_sample")
def get_wav_files(wav_path):
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.sep.join([dirpath, filename])
                if os.stat(filename_path).st_size < 240000:  # 剔除掉一些小文件
                    continue
                wav_files.append(filename_path)
    return wav_files



def data_preprocess(wav_files,number_of_classes):
    data_x = []
    data_y = []
    sample_frequencies=[]
    segment_times=[]
    begin_time = time.time()
    for i, onewav in enumerate(wav_files):
        if i % 5 == 4:  # 运行5个路径名后。
            gaptime = time.time() - begin_time
            percent = float(i) * 100 / len(wav_files)
            eta_time = gaptime * 100 / (percent + 0.01) - gaptime
            strprogress = "[" + "=" * int(percent // 2) + ">" + "-" * int(50 - percent // 2) + "]"
            str_log = ("%.2f %% %s %s/%s \t used:%d s  eta:%d s" % (percent, strprogress, i, len(wav_files), gaptime, eta_time))
            sys.stdout.write('\r' + str_log)

        elements = onewav.split("\\")
        for x in elements:
            if x == 'diode':
                label = 0
            elif x == 'metalnode':
                label = 1
            elif x == 'mouse-wav':
                label =2
            elif x == 'qiangkaiguan-wav':
                label =3
            elif x == 'dianluban-wav':
                label =4


        (rate, data) = wav.read(onewav)
        #注意！考虑到所有音频数据左声道信号非常清晰，而右声道信号很弱很难分辨，因此此处仅采用左声道的数据
        data=np.transpose(data)[0]
        sample_frequency,segment_time,spectrogram=signal.spectrogram(data)
        sample_frequencies.append(sample_frequency)
        segment_times.append(segment_time)
        
        data_x.append(spectrogram)
        data_y.append(label)
    len_freq=[]
    len_time=[]
    for i in sample_frequencies:
        len_freq.append(len(i))
    for i in segment_times:
        len_time.append(len(i))
    #print("\n")
    #print(max(len_freq),min(len_freq),max(len_time),min(len_time))

    #train_x = np.asarray(train_x)
    #train_y = np.asarray(train_y)
    max_freq=max(len_freq)
    max_time=2000
    data_x = [np.concatenate([i, np.zeros(( max_freq, max_time - i.shape[1]))],axis=1) for i in data_x]
    data_x = np.transpose(train_x,axes=(0,2,1))
    data_y = to_categorical(data_y, num_classes=number_of_classes)
    return data_x,data_y

##########################################################################
##########################################################################
train_files = get_wav_files(os.path.join(wav_path,"train"))
test_files = get_wav_files(os.path.join(wav_path,"test"))

train_x,train_y=data_preprocess(train_files,5)
test_x,test_y=data_preprocess(test_files,5)