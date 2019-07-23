"""
技术要点：1、创建基于声谱图的卷积神经网络模型（十分类），本文件为第五版本
        2、三种功能选择：训练并保存模型、评估模型、类别预测
        3、三种训练方法：2d卷积、沿时间卷积、沿频率卷积
        4、添加了绘制准确率和损失值变化曲线的代码；
        5、注释掉早停法代码行；
        6、模型训练回调函数改为logs_loss。
改进方面：音频预处理后再训练
运行结果：300轮训练后，准确率可达到84%
        准确率和损失值曲线效果较好，其他曲线效果不佳
"""
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True #不全部占满显存, 按需分配
session = tf.Session(config=config)
plt.switch_backend('agg')

a = platform.platform()
if "Windows" in a:
    splitchar = "\\"
elif "Linux" in a:
    splitchar = "/"
print('\n', a, '\n')

ROOT_DIR = os.path.abspath('.')
wav_path = os.path.join(ROOT_DIR, "ALL_hd_random")


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
                data = data[j:j + 132400].copy()
                break
        '''
        '''反向取3.5秒：'''
        data = data[-154450:-1].copy()

        sample_frequency, segment_time, spectrogram = signal.spectrogram(data)
        sample_frequencies.append(sample_frequency)
        segment_times.append(segment_time)

        data_x.append(spectrogram)
        data_y.append(label)


    # len_freq = []
    # len_time = []
    # for i in sample_frequencies:
    #     len_freq.append(len(i))
    # for i in segment_times:
    #     len_time.append(len(i))
    #print("\n")
    #print(max(len_freq), min(len_freq), max(len_time), min(len_time))

    # train_x = np.asarray(train_x)
    # train_y = np.asarray(train_y)
    max_freq = sample_frequencies[0]
    max_time = segment_times[0]
    #data_x = [np.concatenate([i, np.zeros((max_freq, max_time - i.shape[1]))], axis=1) for i in data_x]

    aaa=np.shape(data_x[0])
    
    data_x = np.array(data_x)
    print("\n")
    data_x = np.transpose(data_x, axes=(0, 2, 1))
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

    # model.add(Conv2D(filters=16,kernel_size=(3,3), input_shape=(max_time,max_freq,1),activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Conv2D(filters=8,kernel_size=(3,3),activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Conv2D(filters=4,kernel_size=(3,3),activation='relu'))
    # model.add(BatchNormalization())
    # model.add(MaxPool2D(pool_size=(2,2)))
    # model.add(Flatten())
    # #model.add(Dropout(0.5))
    # model.add(Dense(128, activation='relu'))
    # #model.add(Dropout(0.5))
    # model.add(Dense(number_of_classes, activation='softmax'))

    model.add(Conv1D(max_freq, 10, input_shape=(max_time, max_freq), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(4))
    model.add(Conv1D(max_freq, 4, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool1D(4))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(max_freq, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['categorical_accuracy'])


    class LossHistory(keras.callbacks.Callback):
        # 函数开始时创建盛放loss与acc的容器
        def on_train_begin(self, logs={}):
            self.losses = {'batch': [], 'epoch': []}
            self.accuracy = {'batch': [], 'epoch': []}
            self.val_loss = {'batch': [], 'epoch': []}
            self.val_acc = {'batch': [], 'epoch': []}

        # 按照batch来进行追加数据
        def on_batch_end(self, batch, logs={}):
            # 每一个batch完成后向容器里面追加loss，acc
            self.losses['batch'].append(logs.get('loss'))
            self.accuracy['batch'].append(logs.get('acc'))
            self.val_loss['batch'].append(logs.get('val_loss'))
            self.val_acc['batch'].append(logs.get('val_acc'))
            # 每五秒按照当前容器里的值来绘图
            if int(time.time()) % 5 == 0:
                self.draw_p(self.losses['batch'], 'loss', 'train_batch')
                self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
                self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
                self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')

        def on_epoch_end(self, batch, logs={}):
            # 每一个epoch完成后向容器里面追加loss，acc
            self.losses['epoch'].append(logs.get('loss'))
            self.accuracy['epoch'].append(logs.get('acc'))
            self.val_loss['epoch'].append(logs.get('val_loss'))
            self.val_acc['epoch'].append(logs.get('val_acc'))
            # 每五秒按照当前容器里的值来绘图
            if int(time.time()) % 5 == 0:
                self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
                self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
                self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
                self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')

        # 绘图，这里把每一种曲线都单独绘图，若想把各种曲线绘制在一张图上的话可修改此方法
        def draw_p(self, lists, label, type):
            plt.figure()
            plt.plot(range(len(lists)), lists, 'r', label=label)
            plt.ylabel(label)
            plt.xlabel(type)
            plt.legend(loc="upper right")
            plt.savefig(type + '_' + label + '.jpg')

        # 由于这里的绘图设置的是5s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程
        # （最后一次绘图结束，又训练了0-5秒的时间）
        # 所以这里的方法会在整个训练结束以后调用
        def end_draw(self):
            self.draw_p(self.losses['batch'], 'loss', 'train_batch')
            self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
            self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
            self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
            self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
            self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
            self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
            self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')

    logs_loss = LossHistory()


    # model=load_model('voice_recog_spectrogram_new1.h5')
    # print(model.summary())
    # model.pop()
    # model.add(Dense(number_of_classes, activation='softmax',name='output'))
    # model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=[categorical_accuracy])

    # early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.fit(train_x, train_y, batch_size=20, epochs=300, validation_split=0.1, callbacks=[logs_loss])  # callbacks=[early_stopping]
    # 保存模型。
    model.save('voice_recog_spectrogram_preprcsess_300epochs_04.h5')

    logs_loss.end_draw()



    """第一种方法：训练完成时直接绘制acc和loss变化曲线
    train_log = model.fit_generator(train_generator,
                                    steps_per_epoch = nb_train_samples// batch_size,
                                    epochs = epochs,
                                    validation_data = validation_generator,
                                    validation_steps  =nb_validation_samples // batch_size,
                                    )
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), train_log.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), train_log.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), train_log.history["acc"], label="train_acc")
    plt.plot(np.arange(0, epochs), train_log.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on sar classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper right")
    plt.savefig("Loss_Accuracy_alexnet_{:d}e.jpg".format(epochs))
"""



    """第二种方法：训练过程中保留Accuracy和Loss值至csv文件，完成后再读取画图
import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv('./log/mix_r40_g800_log_0511160953_300e.csv')

l = list(log['epoch;acc;loss;val_acc;val_loss'])

epoch = []
acc = []
loss = []
val_acc = []
val_loss = []

for i in range(0,len(l)):
    epoch.append(l[i].split(';')[0])
    acc.append(l[i].split(';')[1])
    loss.append(l[i].split(';')[2])
    val_acc.append(l[i].split(';')[3])
    val_loss.append(l[i].split(';')[4])


plt.style.use("ggplot")                          #设置绘图风格
plt.figure(figsize=(15,10))                      #设置绘图大小，单位inch
plt.plot(epoch, loss, label="train_loss")
plt.plot(epoch, val_loss, label="val_loss")
plt.plot(epoch, acc, label="train_acc")
plt.plot(epoch, val_acc, label="val_acc")
plt.title("Training Loss and Accuracy on sar classifier")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("Loss_Accuracy_mix_40-800_300e.jpg")
"""



elif task == 'evaluate':
    model = load_model('voice_recog_spectrogram_new2.h5')
    accuracy = model.evaluate(test_x, test_y, batch_size=1)
    print('test loss and accuracy:', accuracy)
elif task == 'predict':
    model = load_model('voice_recog_spectrogram_new2.h5')
    result = model.predict_on_batch(test_x)
    print(result)

# from keras.utils.vis_utils import plot_model
# plot_model(model,to_file="model_1.png",show_shapes=True)
