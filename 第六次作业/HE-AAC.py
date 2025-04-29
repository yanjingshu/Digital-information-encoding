import os
import numpy as np
import subprocess
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

def encode_to_heaac(wav_file_path, heaac_file_path, bitrate='48k'):
    """
    使用ffmpeg命令行将wav编码为HE-AAC格式
    """
    cmd = [
        'ffmpeg', '-y', '-i', wav_file_path,
        '-c:a', 'aac', 
        '-b:a', bitrate,
        '-profile:a', 'aac_he',
        heaac_file_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not os.path.exists(heaac_file_path) or os.path.getsize(heaac_file_path) < 1000:
        print("HE-AAC编码失败，请检查ffmpeg是否支持aac_he profile或libfdk_aac。")
        print("ffmpeg输出：")
        print(res.stderr.decode("utf-8"))
        return False
    return True

def decode_to_wav(input_file, output_wav):
    """
    用ffmpeg将m4a/aac解码为wav
    """
    cmd = [
        'ffmpeg', '-y', '-i', input_file, '-ac', '1', '-ar', '44100', output_wav
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not os.path.exists(output_wav):
        print(f"ffmpeg解码失败，未生成{output_wav}，ffmpeg输出如下：")
        print(result.stderr.decode("utf-8"))

def wav_to_heaac(wav_file_path, bitrate='48k'):
    # 读取原始WAV
    sample_rate, original_audio = wavfile.read(wav_file_path)
    if original_audio.ndim == 2:
        original_audio = original_audio[:, 0]  # 只用左声道

    # 输出文件路径
    heaac_file_path = wav_file_path.replace('.wav', f'_heaac_{bitrate}.m4a')
    heaac_decoded_wav = wav_file_path.replace('.wav', f'_heaac_{bitrate}_decoded.wav')

    # 编码为HE-AAC
    if not encode_to_heaac(wav_file_path, heaac_file_path, bitrate):
        return

    # 解码回wav用于对比
    decode_to_wav(heaac_file_path, heaac_decoded_wav)
    if not os.path.exists(heaac_decoded_wav):
        print("解码失败，无法完成SNR计算。")
        return

    try:
        _, heaac_audio = wavfile.read(heaac_decoded_wav)
        if heaac_audio.ndim == 2:
            heaac_audio = heaac_audio[:, 0]
    except Exception as e:
        print(f"HE-AAC解码回wav失败: {e}")
        return

    # 保证对齐
    min_len = min(len(original_audio), len(heaac_audio))
    original_audio = original_audio[:min_len]
    heaac_audio = heaac_audio[:min_len]

    # 计算文件大小和压缩率
    wav_size = os.path.getsize(wav_file_path)
    heaac_size = os.path.getsize(heaac_file_path)
    compression_ratio = wav_size / heaac_size if heaac_size > 0 else 0

    # SNR
    snr = calculate_snr(original_audio, heaac_audio)

    print(f"WAV文件大小: {wav_size/1024:.2f} KB")
    print(f"HE-AAC文件大小: {heaac_size/1024:.2f} KB")
    print(f"压缩率: {compression_ratio:.2f}")
    print(f"SNR: {snr:.2f} dB")
    print("-" * 40)

    # 可选：清理临时文件
    # os.remove(heaac_decoded_wav)

if __name__ == "__main__":
    # 替换为你的WAV文件路径
    wav_file = "sample-1.wav"
    wav_to_heaac(wav_file, bitrate='96k')