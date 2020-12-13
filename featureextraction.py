# import os,sys
# PROGRAM_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))

# sys.path.append(PROGRAM_DIR+ "\\windows\\ffmpeg.exe")
# print(sys.path)
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import matplotlib.pyplot as plt
from numpy import fft
import numpy as np
import pyspark
import time
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw
DEBUG = True


def cov_fft(image_link):  # name
    [Fs, x] = audioBasicIO.read_audio_file(image_link)
    x = audioBasicIO.stereo_to_mono(x)
    F, f_names = ShortTermFeatures.feature_extraction(
        x, Fs, 0.050*Fs, 0.025*Fs)
#     for k in F:
#         norm = np.linalg.norm(k)
#         k = k/norm
    if DEBUG and False:
        for k in range(33):
            plt.subplot(2, 1, 1)
            plt.plot(F[k, :], label=str(k))
            plt.xlabel('Frame no')
            plt.ylabel(f_names[k])

            plt.show()
    F = F / np.linalg.norm(F, axis = 1, keepdims = True)

    return F, fft.fftn(F)


def cal_ifft(arr, ffty, maxcount=5):
    resarr = []
    for item in arr:
        ifftres = np.array([np.real(fft.irfft(p))
                            for p in item * np.conj(ffty)])
        ifftres = ifftres / np.linalg.norm(ifftres, axis = 1, keepdims = True)
        # from scipy.signal import savgol_filter
        # ifftres = savgol_filter(ifftres, 51, 2)

        # box = np.ones(20)/20
        # ifftres = np.array([np.convolve(item, box, mode='full') for item in ifftres])
        if DEBUG:
            # for item in ifftres.copy():
            #     plt.plot(item)
            #     plt.show()
            plt.plot(ifftres.T)
            plt.show()
        zscore = []
        maxidx = np.argmax(ifftres, axis=1)
        for idx, row in zip(maxidx, ifftres):
            zscore.append([idx, (idx - row.mean(axis=0)) / row.std(axis=0)])

        print(sorted(zscore, key=lambda p: p[0]), ifftres.shape)
        cost = ifftres/np.max(ifftres)
        resarr.append(cost)
    return resarr[:maxcount]

def cov_dtw(image_link):  # name
    [Fs, x] = audioBasicIO.read_audio_file(image_link)
    x = audioBasicIO.stereo_to_mono(x)
    F, f_names = ShortTermFeatures.feature_extraction(
        x, Fs, 0.050*Fs, 0.025*Fs)

    if DEBUG and False:
        for k in range(33):
            plt.subplot(2, 1, 1)
            plt.plot(F[k, :], label=str(k))
            plt.xlabel('Frame no')
            plt.ylabel(f_names[k])

            plt.show()
    F = F / np.linalg.norm(F, axis=1, keepdims=True)

    return F


if '__main__' == __name__:
    start_time = time.time()
    F, fff = cov_fft('back.mp3')
    #F = np.random.randn(100, 1)
    print(F.shape, fff.shape)
    plt.plot(F.T)
    plt.show()

    #V, ffv = cov_fft('iyah.mp3')
    # print(V.shape, V.shape)

    #V = V[:,720:1000].copy()
    V = F[:, 999:2000].copy()
    if F.shape[1] > V.shape[1]:
        V = np.concatenate(
            (V,np.zeros((F.shape[0], F.shape[1]-V.shape[1]))), axis=1) #np.random.rand(F.shape[0], F.shape[1]-V.shape[1])), axis=1)
        fff = fft.fftn(F)
        vvv =fft.fftn(V)
    else:
        F = np.concatenate(
            (F, np.zeros((F.shape[0], F.shape[1]-V.shape[1]))), axis=1)
        fff = fft.fftn(F)
        vvv =fft.fftn(V)
    plt.plot(V.T)
    plt.show()
    print("---{}s seconds---".format(time.time()-start_time))
    cal_ifft([fff], vvv)
    print("---{}s seconds---".format(time.time()-start_time))

    distance, path = fastdtw(F.T, V.T, dist=euclidean)
    print(distance)
    plt.plot(path)
    plt.show()