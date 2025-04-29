import os
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile

def calculate_snr(original, compressed):
    """
    计算信噪比（SNR）
    :param original: 原始音频数据（一维数组）
    :param compressed: 压缩后的音频数据（一维数组）
    :return: 信噪比（dB）
    """
    # 转为float防止溢出
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    signal_power = np.sum(np.square(original))
    noise_power = np.sum(np.square(original - compressed))
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def wav_to_mp3_vbr(wav_file_path, vbr_quality=9):
    """
    将WAV文件转换为MP3-VBR格式，并打印压缩率和SNR
    :param wav_file_path: 输入的WAV文件路径
    :param vbr_quality: VBR质量（0为最好，9为最差，推荐2-5）
    """
    # 读取WAV文件
    sample_rate, original_audio = wavfile.read(wav_file_path)
    if original_audio.ndim == 2:
        original_audio = original_audio[:, 0]  # 取左声道

    # 加载并转为单声道
    audio = AudioSegment.from_wav(wav_file_path).set_channels(1)

    # 输出MP3文件名
    mp3_file_path = wav_file_path.replace('.wav', f'_vbr{vbr_quality}.mp3')

    try:
        # 导出为VBR MP3（注意：参数vbr='on'）
        audio.export(mp3_file_path, format='mp3', parameters=["-q:a", str(vbr_quality), "-codec:a", "libmp3lame", "-vn", "-y", "-map_metadata", "-1", "-map_chapters", "-1"])
    except Exception as e:
        print(f"MP3 VBR编码失败: {e}")
        return

    # 读取MP3进行SNR计算
    mp3_audio = AudioSegment.from_mp3(mp3_file_path).set_channels(1)
    mp3_audio_np = np.array(mp3_audio.get_array_of_samples())

    # 保证长度一致
    min_len = min(len(original_audio), len(mp3_audio_np))
    original_audio = original_audio[:min_len]
    mp3_audio_np = mp3_audio_np[:min_len]

    # 计算文件大小与压缩率
    wav_size = os.path.getsize(wav_file_path)
    mp3_size = os.path.getsize(mp3_file_path)
    compression_ratio = wav_size / mp3_size if mp3_size > 0 else 0

    # SNR计算
    snr = calculate_snr(original_audio, mp3_audio_np)

    # 输出结果
    print(f"WAV文件大小: {wav_size/1024:.2f} KB")
    print(f"MP3-VBR文件大小: {mp3_size/1024:.2f} KB")
    print(f"压缩率: {compression_ratio:.2f}")
    print(f"SNR: {snr:.2f} dB")
    print("-" * 40)

if __name__ == "__main__":
    # 替换为你的WAV文件路径
    wav_file = "sample-5.wav"
    # VBR质量参数可选：0（最好）~9（最差），推荐用2-5
    wav_to_mp3_vbr(wav_file, vbr_quality=9)