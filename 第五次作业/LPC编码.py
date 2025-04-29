import numpy as np
import librosa
import soundfile as sf
from scipy.linalg import solve_toeplitz
import os
import time

def lpc(signal, order):
    """LPC系数计算"""
    n = len(signal)
    r = np.array([np.sum(signal[:n - k] * signal[k:]) for k in range(order + 1)])
    a = solve_toeplitz(r[:-1], -r[1:])
    return np.concatenate(([1], a))

def generate_robot_voice(signal, order=12, frame_length=1024, hop_length=512):
    """帧级LPC编码和机器人音生成"""
    robot_voice = np.zeros_like(signal)
    num_frames = int((len(signal) - frame_length) / hop_length) + 1
    times = []
    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frame = signal[start:end]
        if np.all(frame == 0):  # 跳过静音帧
            continue
        
        t0 = time.perf_counter()
        # LPC分析
        a = lpc(frame, order)
        # 机器人声源：白噪声
        excitation = np.random.randn(frame_length)
        # 合成
        synth = np.zeros_like(frame)
        for n in range(order, frame_length):
            synth[n] = excitation[n] - np.sum(a[1:] * synth[n-order:n][::-1])
        t1 = time.perf_counter()
        # 保存到输出
        robot_voice[start:end] += synth
        times.append(t1 - t0)
    return robot_voice, times

def calculate_snr(original, processed):
    """信噪比计算"""
    noise = original - processed[:len(original)]
    signal_power = np.sum(original ** 2)
    noise_power = np.sum(noise ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr

def calculate_compression_ratio(input_file, output_file):
    """压缩率计算"""
    input_size = os.path.getsize(input_file)
    output_size = os.path.getsize(output_file)
    if output_size == 0:
        return 0
    return input_size / output_size

def wav_to_lpc_mp3(input_wav, output_mp3, order=12, frame_length=1024, hop_length=512):
    signal, sr = librosa.load(input_wav, sr=None)
    if signal.ndim > 1:
        signal = librosa.to_mono(signal)
    # 生成机器人音
    robot_voice, times = generate_robot_voice(signal, order, frame_length, hop_length)
    # 归一化音量
    robot_voice = robot_voice / (np.max(np.abs(robot_voice)) + 1e-10) * 0.99
    sf.write(output_mp3, robot_voice.astype(np.float32), sr, format='MP3')
    print(f"LPC编码后已保存为: {output_mp3}")

    # 计算SNR
    min_len = min(len(signal), len(robot_voice))
    snr = calculate_snr(signal[:min_len], robot_voice[:min_len])
    print(f"SNR: {snr:.2f} dB")

    # 计算压缩率
    compression_ratio = calculate_compression_ratio(input_wav, output_mp3)
    print(f"压缩率: {compression_ratio:.2f}")

    # 单帧处理时间
    if times:
        print(f"平均单帧处理时间: {np.mean(times)*1000:.2f} ms, 最大: {np.max(times)*1000:.2f} ms, 最小: {np.min(times)*1000:.2f} ms, 帧数: {len(times)}")
    else:
        print("没有有效帧进行处理。")

if __name__ == "__main__":
    input_wav = "example.wav"      # 输入wav文件路径
    output_mp3 = "lpc_output.mp3"  # 输出mp3文件路径
    wav_to_lpc_mp3(input_wav, output_mp3)