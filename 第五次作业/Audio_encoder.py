# 导入所需库
import numpy as np
import matplotlib.pyplot as plt
import librosa
from pydub import AudioSegment


# 音频频谱分析函数
def audio_spectrum_analysis(audio_file):
    # 加载音频文件
    signal, sr = librosa.load(audio_file, sr=None)
    if signal.ndim > 1:
        signal = librosa.to_mono(signal)

    # 进行FFT变换
    fft = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / sr)

    # 绘制频谱图
    plt.figure(figsize=(10, 4))
    plt.plot(freqs[:len(freqs) // 2], np.abs(fft)[:len(freqs) // 2])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Spectrum of the Audio Signal')
    plt.grid()
    plt.show()

    # 提取频谱的主要特征
    peak_freq = freqs[np.argmax(np.abs(fft))]
    peak_mag = np.max(np.abs(fft))
    print(f"Peak Frequency: {peak_freq} Hz")
    print(f"Peak Magnitude: {peak_mag}")


# 音频编码函数（MP3 编码）
def audio_encode_mp3(input_file, output_file, bitrate="128k"):
    # 加载音频文件
    audio = AudioSegment.from_file(input_file)

    # 导出为 MP3 文件
    audio.export(output_file, format="mp3", bitrate=bitrate)
    print(f"Audio encoded and saved to {output_file}")


# 主函数
if __name__ == "__main__":
    # 音频频谱分析
    audio_file = 'example.wav'  # 替换为你的音频文件路径
    audio_spectrum_analysis(audio_file)

    # 音频编码（MP3 编码）
    input_file = 'example.wav'  # 替换为你的音频文件路径
    output_file = 'output.mp3'  # 替换为你想要保存的编码后文件路径
    audio_encode_mp3(input_file, output_file)