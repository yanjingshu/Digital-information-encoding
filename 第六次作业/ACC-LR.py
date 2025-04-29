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
    original = original.astype(np.float64)
    compressed = compressed.astype(np.float64)
    signal_power = np.sum(np.square(original))
    noise_power = np.sum(np.square(original - compressed))
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def wav_to_aac_lc(wav_file_path, bitrate='128k'):
    """
    将WAV文件转换为AAC_LC格式，并打印压缩率和SNR
    :param wav_file_path: 输入的WAV文件路径
    :param bitrate: AAC编码比特率，默认128k
    """
    # 读取WAV文件
    sample_rate, original_audio = wavfile.read(wav_file_path)
    if original_audio.ndim == 2:
        original_audio = original_audio[:, 0]  # 取左声道

    # 加载并转为单声道
    audio = AudioSegment.from_wav(wav_file_path).set_channels(1)

    # 输出AAC文件名
    aac_file_path = wav_file_path.replace('.wav', f'_aac_{bitrate}.m4a')

    try:
        # 使用 ffmpeg 的 AAC_LC 编码器，写为 .m4a 容器
        audio.export(
            aac_file_path,
            format='ipod',  # ipod = m4a = AAC_LC
            bitrate=bitrate,
            parameters=["-profile:a", "aac_low"]
        )
    except Exception as e:
        print(f"AAC_LC编码失败: {e}")
        return

    # 读取AAC进行SNR计算
    aac_audio = AudioSegment.from_file(aac_file_path).set_channels(1)
    aac_audio_np = np.array(aac_audio.get_array_of_samples())

    # 保证长度一致
    min_len = min(len(original_audio), len(aac_audio_np))
    original_audio = original_audio[:min_len]
    aac_audio_np = aac_audio_np[:min_len]

    # 计算文件大小与压缩率
    wav_size = os.path.getsize(wav_file_path)
    aac_size = os.path.getsize(aac_file_path)
    compression_ratio = wav_size / aac_size if aac_size > 0 else 0

    # SNR计算
    snr = calculate_snr(original_audio, aac_audio_np)

    # 输出结果
    print(f"WAV文件大小: {wav_size/1024:.2f} KB")
    print(f"AAC_LC文件大小: {aac_size/1024:.2f} KB")
    print(f"压缩率: {compression_ratio:.2f}")
    print(f"SNR: {snr:.2f} dB")
    print("-" * 40)

if __name__ == "__main__":
    # 替换为你的WAV文件路径
    wav_file = "sample-5.wav"
    # 你可以修改比特率如'96k', '128k', '192k'等
    wav_to_aac_lc(wav_file, bitrate='128k')