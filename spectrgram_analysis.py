import numpy as np
from scipy import signal
import scipy.io.wavfile as wav
import os
import time
import sys
from keras.utils.np_utils import to_categorical
#import matplotlib.pyplot as plt
#import skimage.io
import platform

a=platform.platform()
if "Windows" in a:
    splitchar="\\"
elif "Linux" in a:
    splitchar="/"
print(a)


ROOT_DIR=os.path.abspath('.')
wav_path = os.path.join(ROOT_DIR,"ALL_hd_random")
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

        elements = onewav.split(splitchar)
        for x in elements:
            if x == '01 diode':
                label = 0
            elif x == '02 metalnode':
                label = 1
            elif x == '03 qiangkaiguan':
                label =2
            elif x == '04 mouse':
                label =3
            elif x == '05 dianluban':
                label =4
            elif x== '06 libattery':
                label =5
            elif x== '07 charger':
                label =6
            elif x== '08 A-wav':
                label =7
            elif x== '09 B-wav':
                label =8
            elif x== '10 C-wav':
                label =9
            elif x== '11 D-wav':
                label =10
            elif x== '12 E-wav':
                label =11
            elif x== '13 F-wav':
                label =12
            elif x== '14 A+mouse':
                label = 13
            elif x== '15 C+libattery':
                label=14
            elif x== '16 D+charger':
                label=15

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
    data_x = np.transpose(data_x,axes=(0,2,1))
    data_y = to_categorical(data_y, num_classes=number_of_classes)
    return data_x,data_y,max_freq,max_time

##########################################################################
##########################################################################

number_of_classes=16

#读取文件
train_files = get_wav_files(os.path.join(wav_path,"train"))
test_files = get_wav_files(os.path.join(wav_path,"test"))

#数据预处理
train_x,train_y,max_freq,max_time=data_preprocess(train_files,number_of_classes)
test_x,test_y,max_freq,max_time=data_preprocess(test_files,number_of_classes)


from keras.models import Sequential,load_model
from keras.layers import Conv1D,MaxPool1D,Conv2D,MaxPool2D,Flatten,Dense,BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.metrics import categorical_accuracy


task='predict' #train or evaluate or predict
if task=='train':
    #2d卷积模型
    # model = Sequential()
    # model.add(Conv2D(filters=64, kernel_size=[5,5], input_shape=(max_freq, max_time,1)))
    # model.add(MaxPool2D([3,3]))
    # model.add(Conv2D(32, [3,3]))
    # model.add(MaxPool2D([3,3]))
    # model.add(Conv2D(16, [3,3]))
    # model.add(Flatten())
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(2, activation='softmax'))
    # model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=[categorical_accuracy])

    #沿时间卷积模型
    model = Sequential()
    model.add(Conv1D(max_freq, 10, input_shape=(max_time,max_freq),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(4))
    model.add(Conv1D(max_freq, 4,activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(4))
    model.add(Flatten())
    model.add(Dense(max_freq, activation='relu'))
    model.add(Dense(number_of_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=[categorical_accuracy])

    #沿频率卷积模型
    # model = Sequential()
    # model.add(Conv1D(700, 3, input_shape=(max_freq,max_time),activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPool1D(3))
    # model.add(Conv1D(350, 3,activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPool1D(3))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(2, activation='softmax')) 
    # model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=[categorical_accuracy])

    model.fit(train_x, train_y,batch_size=10, epochs=20)


    # 保存模型。
    model.save('voice_recog_spectrogram_new1.h5')
    print(model.summary())

elif task=='evaluate':
    model=load_model('voice_recog_spectrogram_new1.h5')
    accuracy = model.evaluate(test_x, test_y, batch_size=1)
    print('test loss and accuracy:',accuracy)
elif task=='predict':
    model=load_model('voice_recog_spectrogram_new1.h5')
    result=model.predict_on_batch(test_x)
    print(result)


# from keras.utils.vis_utils import plot_model
# plot_model(model,to_file="model_1.png",show_shapes=True)