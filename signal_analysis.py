'''''''''''''''''''''''''''
提取一个音频文件的spectrogram并使用matplotlib或者skimage可视化（划掉
提取一个音频文件的各种图并使用matplotlib可视化
'''''''''''''''''''''''''''
import numpy as np
from scipy import signal
import scipy.io.wavfile as wav
import os
import matplotlib.pyplot as plt
import skimage.io
from scipy.fftpack import fft

ROOT_DIR=os.path.abspath('.')
# # 旧，diode
# example_file=os.path.join(ROOT_DIR,"hd_signal_keras_train","diode","diade (1).wav")
# # 旧，metalnode
# example_file=os.path.join(ROOT_DIR,"hd_signal_keras_train","metalnode","metalnode (1).wav")
# # 新，diode
# example_file=os.path.join(ROOT_DIR,"signal_2","diode-wav","diode (1).wav")
# # 新，metalnode
example_file=os.path.join(ROOT_DIR,"signal_2","metalnode-wav","metalnode (1).wav")

simple_rate,data=wav.read(example_file)
# print(simple_rate)
# print(data)
data=np.transpose(data)[0]  #只取左声道


#功率谱密度
f, Pwelch_spec = signal.welch(data, fs=simple_rate, scaling='spectrum')
plt.semilogy(f, Pwelch_spec)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD')
plt.grid()
plt.show()

#画频谱图
fft_size=131072 #FFT处理的取样长度
win=signal.windows.tukey(fft_size,0.25)

yf=fft(data[:fft_size])
yf = np.abs(yf*win)/fft_size*2
yf=yf[1:fft_size//2]
freqs = np.linspace(0, simple_rate/2, fft_size/2)
freqs=freqs[1:fft_size//2]
# xfp = 20*np.log10(np.clip(xf, 1e-20, 1e100))

plt.plot(freqs,yf)
plt.show()


#画波形图
time=np.arange(0,len(data))
plt.plot(time,data)
plt.show()

# from pylab import specgram
# #与spectrogram相比，该处自动计算了20*np.log10()，大概吧
# specgram(data, NFFT=256, Fs=1, noverlap=32)
# # plt.show()

#画声谱图
sample_frequencies,segment_times,spectrogram=signal.spectrogram(data,fs=simple_rate,nperseg=256) #提取spectrogram的参数均使用默认值
# print(spectrogram)
# print(np.min(spectrogram)) #最小值为0，可画图

from matplotlib.ticker import FuncFormatter
fig, ax = plt.subplots(1, 1)
ax0=ax.imshow(spectrogram)       
def formatnum(y, pos):
    return '$%d$' % (y*simple_rate/spectrogram.shape[0]/2)
formatter = FuncFormatter(formatnum)
ax.yaxis.set_major_formatter(formatter)
# ax.set_ylim(0,32)
fig.colorbar(ax0,ax=ax)
plt.show()

# skimage.io.imshow(spectrogram)
# skimage.io.show()

