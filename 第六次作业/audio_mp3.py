import os
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile

def calculate_snr(original, compressed):
    """
    计算信噪比（SNR）
    :param original: 原始音频数据（一维数组）
    :param compressed: 压缩后的音频数据（一维数组）
    :return: 信噪比
    """
    signal_power = np.sum(np.square(original.astype(np.float64)))
    noise_power = np.sum(np.square(original.astype(np.float64) - compressed.astype(np.float64)))
    if noise_power == 0:
        return float('inf')
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def wav_to_mp3(wav_file_path, bitrate):
    """
    将 WAV 文件转换为指定码率的 MP3 文件，并计算 SNR 和压缩率
    :param wav_file_path: 输入的 WAV 文件路径
    :param bitrate: 目标 MP3 文件的码率，如 '128k'
    """
    # 读取 WAV 文件
    sample_rate, original_audio = wavfile.read(wav_file_path)
    # 如果是立体声，取左声道
    if original_audio.ndim == 2:
        original_audio = original_audio[:, 0]

    # 加载 WAV 音频
    audio = AudioSegment.from_wav(wav_file_path)
    # 转为单声道以避免声道不一致
    audio = audio.set_channels(1)

    # 生成 MP3 文件路径
    mp3_file_path = wav_file_path.replace('.wav', f'_{bitrate}.mp3')

    try:
        # 导出为指定码率的 MP3 格式
        audio.export(mp3_file_path, format='mp3', bitrate=bitrate)
    except Exception as e:
        print(f"MP3 编码失败: {e}")
        return

    # 读取编码后的 MP3 文件
    mp3_audio = AudioSegment.from_mp3(mp3_file_path)
    mp3_audio = mp3_audio.set_channels(1)
    mp3_audio = np.array(mp3_audio.get_array_of_samples())

    # 保证长度一致
    min_len = min(len(original_audio), len(mp3_audio))
    original_audio = original_audio[:min_len]
    mp3_audio = mp3_audio[:min_len]

    # 获取文件大小
    wav_size = os.path.getsize(wav_file_path)
    mp3_size = os.path.getsize(mp3_file_path)

    # 计算压缩率
    compression_ratio = wav_size / mp3_size

    # 计算 SNR
    snr = calculate_snr(original_audio, mp3_audio)

    # 打印结果
    print(f"原始 WAV 文件大小: {wav_size / 1024:.2f} KB")
    print(f"MP3 文件（码率 {bitrate}）大小: {mp3_size / 1024:.2f} KB")
    print(f"压缩率: {compression_ratio:.2f}")
    print(f"信噪比 (SNR): {snr:.2f} dB")
    print("-" * 50)

if __name__ == "__main__":
    # 请替换为你的 WAV 文件路径
    wav_file = "sample-5.wav"
    # 定义不同的码率
    bitrates = ['64k', '128k', '192k', '320k']
    for bitrate in bitrates:
        wav_to_mp3(wav_file, bitrate)