import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from numpy.fft import rfft, rfftfreq
from tqdm import trange
from scipy.signal import welch

fs = 20_000           # 采样频率 20 kHz
f1 = 50               # 基波 50 Hz
duration = 2          # 采 2 秒（100 个周波）
t = np.arange(0, duration, 1 / fs)

A1 = 1.0              # 基波峰值
A3 = 0.1              # 三次谐波峰值（10%）
signal = (A1 * np.sin(2 * np.pi * f1 * t) +
          A3 * np.sin(2 * np.pi * 3 * f1 * t))
thd_vals = []

fs = 20000  # 采样频率, 例 20 kHz
freqs, psd = welch(signal, fs=fs, nperseg=len(signal), window='boxcar')

# 找离 50 Hz 最近的频点作为基波
f1_idx = np.argmin(np.abs(freqs - 50))
# 基波 RMS = PSD(f1) × 带宽，再开方
bandwidth = freqs[1] - freqs[0]  # 相邻频点间隔
fundamental_rms = np.sqrt(psd[f1_idx] * bandwidth)

total_rms = np.sqrt(np.trapz(psd, freqs))  # PSD 积分求总体 RMS
harm_rms = np.sqrt(max(total_rms ** 2 - fundamental_rms ** 2, 0.0))
# 加 1e-12 防 0 除
thd_vals.append(harm_rms / (fundamental_rms + 1e-12))

print(thd_vals)
