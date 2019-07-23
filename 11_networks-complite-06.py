"""
技术要点：1、创建基于声谱图的卷积神经网络模型（十分类），本文件为第六版本
        2、三种功能选择：训练并保存模型、评估模型、类别预测
        3、采用声谱图沿时间一维卷积的训练方法
        4、添加了绘制准确率和损失值变化曲线的代码
改进方面：音频预处理,末端对齐反向取3秒时长后再训练
        lr降低率0.8为最佳，batch_size设置8为最佳
运行结果：300轮训练后，准确率可达到84%，损失值可降至1.16
        准确率和损失值曲线效果显示良好，均趋于稳定，无震荡
下步可做：为进一步提高准确率，在此基础上，下一步可以加深网络结构，或减小样本大小。
"""
# import pandas as pd
import numpy as np
from scipy import signal
import scipy.io.wavfile as wav
import os
import time
import sys
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
# import skimage.io
import platform
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
# session = tf.Session(config=config)

plt.switch_backend('agg')

a = platform.platform()
if "Windows" in a:
    splitchar = "\\"
elif "Linux" in a:
    splitchar = "/"
print('\n', a, '\n')

ROOT_DIR = os.path.abspath('.')
wav_path = os.path.join(ROOT_DIR, "ALL_hd_random（10）")


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


def data_preprocess(wav_files, number_of_classes):
    data_x = []
    data_y = []
    sample_frequencies = []
    segment_times = []
    begin_time = time.time()
    for i, onewav in enumerate(wav_files):
        if i % 5 == 4:  # 运行5个路径名后。
            gaptime = time.time() - begin_time
            percent = float(i) * 100 / len(wav_files)
            eta_time = gaptime * 100 / (percent + 0.01) - gaptime
            strprogress = "[" + "=" * int(percent // 2) + ">" + "-" * int(50 - percent // 2) + "]"
            str_log = ("%.2f %% %s %s/%s \t used:%d s  eta:%d s" % (
                percent, strprogress, i, len(wav_files), gaptime, eta_time))
            sys.stdout.write('\r' + str_log)

        elements = onewav.split(splitchar)
        for x in elements:
            if x == '01 diode':
                label = 0
            elif x == '02 metalnode':
                label = 1
            elif x == '03 qiangkaiguan':
                label = 2
            elif x == '04 mouse':
                label = 3
            elif x == '05 dianluban':
                label = 4
            elif x == '06 libattery':
                label = 5
            elif x == '07 charger':
                label = 6
            elif x == '08 A-wav':
                label = 7
            elif x == '09 qiangchazuo':
                label = 8
            elif x == '10 netport':
                label = 9

        (rate, data) = wav.read(onewav)
        # 注意！考虑到所有音频数据左声道信号非常清晰，而右声道信号很弱很难分辨，因此此处仅采用左声道的数据
        data = np.transpose(data)[0]
        '''正向取3秒：
        for j in range(len(data)):  # len(aud)是统计出二元数组aud的行数，len(aud[0])则是统计数组列数。如果多维，每一维是len(A[i])。
            if data[j] > 10 or data[j] < -10:
                data = data[j:(j + 132400)].copy()
                break
        '''
        '''反向取3秒：132400'''
        data = data[-132400:-1].copy()

        sample_frequency, segment_time, spectrogram = signal.spectrogram(data)
        sample_frequencies.append(sample_frequency)
        segment_times.append(segment_time)

        data_x.append(spectrogram)
        data_y.append(label)

    max_freq = len(sample_frequencies[0])
    max_time = len(segment_times[0])
    # data_x = [np.concatenate([i, np.zeros((max_freq, max_time - i.shape[1]))], axis=1) for i in data_x]

    data_x = np.asarray(data_x)
    # data_x = np.transpose(data_x, axes=(0, 2, 1))
    # data_x=np.expand_dims(data_x,axis=3)
    data_y = to_categorical(data_y, num_classes=number_of_classes)
    return data_x, data_y, max_freq, max_time


##########################################################################
##########################################################################

number_of_classes = 10

# 读取文件
train_files = get_wav_files(os.path.join(wav_path, "train"))
test_files = get_wav_files(os.path.join(wav_path, "test"))

# 数据预处理
train_x, train_y, max_freq, max_time = data_preprocess(train_files, number_of_classes)
test_x, test_y, max_freq, max_time = data_preprocess(test_files, number_of_classes)

import random

randnum = random.randint(0, 100)
random.seed(randnum)
random.shuffle(train_x)
random.seed(randnum)
random.shuffle(train_y)

from keras.models import Sequential, load_model
from keras.layers import MaxPool1D, Conv1D, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.metrics import categorical_accuracy
from keras import regularizers
import keras

task = 'train'  # train or evaluate or predict
if task == 'train':
    model = Sequential()

    model.add(Conv1D(filters=max_time, kernel_size=5, padding='same',
                     input_shape=(max_freq, max_time), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(4))

    model.add(Conv1D(filters=max_time, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(4))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(max_time, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(number_of_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['categorical_accuracy'])

    print(model.summary())

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, verbose=0, mode='min',
                                  epsilon=0.001, cooldown=0, min_lr=0)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=50) # verbose=0,mode='auto'
    train_history = model.fit(train_x, train_y, batch_size=8, epochs=300,
                              validation_split=0.1, callbacks=[reduce_lr])  #
    # 保存模型。
    model.save('voice_recog_spectrogram_preprcsess_10lei_1d_300epochs_01.h5')

    fig = plt.figure()  # 新建一张图。plt.plot()
    plt.plot(train_history.history['categorical_accuracy'], label='training acc')
    plt.plot(train_history.history['val_categorical_accuracy'], label='val acc')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    fig.savefig('01-train-val-' + 'acc.png')
    # plt.show()
    # plt.plot()
    fig = plt.figure()
    plt.plot(train_history.history['loss'], label='training loss')
    plt.plot(train_history.history['val_loss'], label='val loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    fig.savefig('02-train-val-' + 'loss.png')
    # plt.show()



elif task == 'evaluate':
    model = load_model('voice_recog_spectrogram_new2.h5')
    accuracy = model.evaluate(test_x, test_y, batch_size=1)
    print('test loss and accuracy:', accuracy)
elif task == 'predict':
    model = load_model('voice_recog_spectrogram_new2.h5')
    result = model.predict_on_batch(test_x)
    print(result)
