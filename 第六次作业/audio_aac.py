import os
import numpy as np
import subprocess
from scipy.io import wavfile

def audiosegment_to_np(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    # pydub's get_array_of_samples returns a flat array, so handle channels:
    if audio_segment.channels > 1:
        samples = samples.reshape((-1, audio_segment.channels))
        samples = samples.mean(axis=1)
    # 16-bit PCM normalization
    return samples.astype(np.float32) / (2**15)

def wav_to_float32_mono(wav_audio):
    # If multi-channel, convert to mono by averaging
    if len(wav_audio.shape) > 1:
        wav_audio = wav_audio.mean(axis=1)
    # Normalize based on dtype
    if wav_audio.dtype == np.int16:
        wav_audio = wav_audio.astype(np.float32) / (2**15)
    elif wav_audio.dtype == np.int32:
        wav_audio = wav_audio.astype(np.float32) / (2**31)
    else:
        wav_audio = wav_audio.astype(np.float32)
    return wav_audio

def calculate_snr(original, compressed):
    # Ensure both signals are float32 and normalized
    original = original.astype(np.float32)
    compressed = compressed.astype(np.float32)
    # Align length
    min_len = min(len(original), len(compressed))
    original = original[:min_len]
    compressed = compressed[:min_len]
    # (Optional) Normalize both signals to max abs 1
    if np.max(np.abs(original)) > 0:
        original = original / np.max(np.abs(original))
    if np.max(np.abs(compressed)) > 0:
        compressed = compressed / np.max(np.abs(compressed))
    # SNR calculation
    noise = original - compressed
    signal_power = np.sum(np.square(original))
    noise_power = np.sum(np.square(noise))
    if noise_power == 0:
        return float('inf')
    if signal_power == 0:
        return 0
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def wav_to_aac(wav_file_path):
    if not os.path.isfile(wav_file_path):
        print(f"文件 {wav_file_path} 不存在")
        return

    # Load WAV as mono, normalized float32
    sample_rate, original_audio = wavfile.read(wav_file_path)
    original_audio = wav_to_float32_mono(original_audio)

    # Encode to AAC
    aac_file_path = wav_file_path.replace('.wav', '.aac')
    try:
        command = ['ffmpeg', '-y', '-f', 'wav', '-i', wav_file_path, '-c:a', 'aac', aac_file_path]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"AAC 编码失败: {e}")
        return

    # Read AAC using pydub
    from pydub import AudioSegment
    try:
        aac_audio_seg = AudioSegment.from_file(aac_file_path, format='aac')
        aac_audio = audiosegment_to_np(aac_audio_seg)
    except Exception as e:
        print(f"读取 AAC 文件失败: {e}")
        return

    # File sizes
    wav_size = os.path.getsize(wav_file_path)
    aac_size = os.path.getsize(aac_file_path)
    compression_ratio = wav_size / aac_size if aac_size != 0 else float('inf')

    # SNR
    snr = calculate_snr(original_audio, aac_audio)

    print(f"原始 WAV 文件大小: {wav_size / 1024:.2f} KB")
    print(f"AAC 文件大小: {aac_size / 1024:.2f} KB，压缩率: {compression_ratio:.2f}，SNR: {snr:.2f} dB")

if __name__ == "__main__":
    # 替换为你的 WAV 文件路径
    wav_file = "sample-5.wav"
    wav_to_aac(wav_file)