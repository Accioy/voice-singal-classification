import numpy as np
from scipy import signal
import scipy.io.wavfile as wav
import os
import time
import sys
import platform
#import matplotlib.pyplot as plt
#import skimage.io

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

wav_files = get_wav_files(wav_path)


train_x = []
train_y = []
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
    
    train_x.append(spectrogram)
    train_y.append(label)
###################################
number_of_classes=5
######################################
len_freq=[]
len_time=[]
for i in sample_frequencies:
    len_freq.append(len(i))
for i in segment_times:
    len_time.append(len(i))
# print("\n")
# print(max(len_freq),min(len_freq),max(len_time),min(len_time))
# #结果：129 129 1429 833
max_time=max(len_time)
max_freq=max(len_freq) #其实频谱图纵坐标长度（即len_freq）都是一样的

#补零使得所有样本shape相同
#train_x = [np.concatenate([i, np.zeros(( max_freq, max_time - i.shape[1]))],axis=1) for i in train_x]
#为兼容其他数据，设定max_time=2000
max_time=2000
train_x = [np.concatenate([i, np.zeros(( max_freq, max_time - i.shape[1]))],axis=1) for i in train_x]
train_x = np.asarray(train_x)

# train_x = train_x[:,:,:,np.newaxis]
train_x = np.transpose(train_x,axes=(0,2,1))

from keras.utils.np_utils import to_categorical
train_y=to_categorical(train_y, num_classes=5)  # (201,2)
train_y = np.asarray(train_y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0,shuffle=True,stratify=train_y)
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


from keras.models import Sequential,load_model,Model
from keras.layers import Conv1D,MaxPool1D,Conv2D,MaxPool2D,Flatten,Dense,BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.metrics import categorical_accuracy






task='evaluate' #train or evaluate or predict
if task=='train':
    model=load_model('voice_recog_spectrogram.h5')
    # layer_name='dense_1'
    # model = Model(inputs=model.input,
    #                              outputs=model.get_layer(layer_name).output)
    model.pop()
    model.add(Dense(number_of_classes, activation='softmax',name='output'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=[categorical_accuracy])
    model.fit(x_train, y_train,batch_size=10, epochs=20)
    # 保存模型。
    model.save('voice_recog_spectrogram_new.h5')
    print(model.summary())

elif task=='evaluate':
    model=load_model('voice_recog_spectrogram_new.h5')
    accuracy = model.evaluate(x_test, y_test, batch_size=1)
    print('test loss and accuracy:',accuracy)
elif task=='predict':
    model=load_model('voice_recog_spectrogram_new.h5')
    result=model.predict_on_batch(x_test)
    print(result)


# from keras.utils.vis_utils import plot_model
# plot_model(model,to_file="model_1.png",show_shapes=True)