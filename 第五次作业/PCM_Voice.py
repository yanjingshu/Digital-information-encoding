import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.io.wavfile import write
import os

# 加载音频文件
audio_file = 'example.wav'  # 替换为你的音频文件路径
signal, sr = librosa.load(audio_file, sr=None)
if signal.ndim > 1:
    signal = librosa.to_mono(signal)

# 进行PCM编码
def pcm_encode(signal, bits):
    max_val = 2**(bits-1) - 1
    min_val = -2**(bits-1)
    scaled_signal = np.int16(signal / np.max(np.abs(signal)) * max_val)
    return scaled_signal

# 设置位数
bits = 16
pcm_signal = pcm_encode(signal, bits)

# 保存PCM编码后的音频
write('pcm_encoded.wav', sr, pcm_signal)

# 计算SNR
def calculate_snr(original, reconstructed):
    noise = original - (reconstructed / np.max(np.abs(reconstructed)) * np.max(np.abs(original)))
    snr = 10 * np.log10(np.sum(original**2) / np.sum(noise**2))
    return snr

snr = calculate_snr(signal, pcm_signal)

# 计算压缩率
original_size = os.path.getsize(audio_file)
compressed_size = os.path.getsize('pcm_encoded.wav')
compression_ratio = original_size / compressed_size

# 打印指标
print(f"SNR (dB): {snr}")
print(f"压缩率: {compression_ratio}")

# 绘制原始信号和PCM编码后的信号
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(signal)
plt.title('Original Signal')
plt.subplot(2, 1, 2)
plt.plot(pcm_signal)
plt.title('PCM Encoded')
plt.tight_layout()
plt.show()