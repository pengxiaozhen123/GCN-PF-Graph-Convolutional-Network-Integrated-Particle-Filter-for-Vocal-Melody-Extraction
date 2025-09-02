import numpy as np
import librosa

wave = r'D:\数据集\测试用\48000\train03.wav'
y, sr = librosa.load(path=wave, sr=48000)
features12 = np.abs(librosa.cqt(y, sr=sr, hop_length=480, fmin=36, n_bins=70, bins_per_octave=12, tuning=0.0,
                filter_scale=1, norm=1, sparsity=0.00, window='hann', scale=True))
features36 = np.abs(librosa.cqt(y, sr=sr, hop_length=480, fmin=36, n_bins=70, bins_per_octave=36, tuning=0.0,
                filter_scale=1, norm=1, sparsity=0.00, window='hann', scale=True))
np.savetxt(r"D:\数据集\测试用\12.csv", features12.T, delimiter=',', fmt="%.3f")
np.savetxt(r"D:\数据集\测试用\36.csv", features36.T, delimiter=',', fmt="%.3f ")
print(librosa.hz_to_midi(440))