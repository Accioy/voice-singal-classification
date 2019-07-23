import numpy as np
from scipy import signal
import scipy.io.wavfile as wav
import os
import time
import sys
#import matplotlib.pyplot as plt
#import skimage.io

ROOT_DIR=os.path.abspath('.')
wav_path = os.path.join(ROOT_DIR,"signal_2")
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

wav_files = get_wav_files(wav_path)


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
        if x == 'diode-wav':
            label = 0
        elif x == 'metalnode-wav':
            label = 1


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

# #结果：129 129 1429 833
max_time=max(len_time)
max_freq=max(len_freq) #其实频谱图纵坐标长度（即len_freq）都是一样的

#补零使得所有样本shape相同
max_time=2000
data_x = [np.concatenate([i, np.zeros(( max_freq, max_time - i.shape[1]))],axis=1) for i in data_x]
data_x = np.asarray(data_x)

# data_x = data_x[:,:,:,np.newaxis]
data_x = np.transpose(data_x,axes=(0,2,1))

from keras.utils.np_utils import to_categorical
data_y=to_categorical(data_y, num_classes=2)  # (201,2)
data_y = np.asarray(data_y)

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


from keras.models import Sequential,load_model
from keras.layers import Conv1D,MaxPool1D,Conv2D,MaxPool2D,Flatten,Dense,BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.metrics import categorical_accuracy


task='evaluate' #evaluate or predict
if task=='evaluate':
    model=load_model('voice_recog_spectrogram.h5')
    accuracy = model.evaluate(data_x, data_y, batch_size=1)
    print('test loss and accuracy:',accuracy)
elif task=='predict':
    model=load_model('voice_recog_spectrogram.h5')
    result=model.predict_on_batch(data_x)
    print(result)


# from keras.utils.vis_utils import plot_model
# plot_model(model,to_file="model_1.png",show_shapes=True)