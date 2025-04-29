import numpy as np
import soundfile as sf
import os
import time

def adpcm9_encode(signal):
    delta = 1.0 / (2**8)  # 步长
    prev = 0
    encoded = []
    for sample in signal:
        diff = sample - prev
        quantized = int(np.round(diff / delta))
        quantized = np.clip(quantized, -256, 255)  # 9位：-256~255
        encoded.append(quantized)
        prev += quantized * delta
    return np.array(encoded, dtype=np.int16)

def adpcm9_decode(encoded):
    delta = 1.0 / (2**8)
    prev = 0
    decoded = []
    for q in encoded:
        prev += q * delta
        decoded.append(prev)
    return np.array(decoded, dtype=np.float32)

def compute_snr(original, reconstructed):
    noise = original - reconstructed
    snr = 10 * np.log10(np.sum(original ** 2) / np.sum(noise ** 2))
    return snr

def add_noise(signal, snr_db):
    """将具有特定SNR的高斯白噪声添加到信号"""
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)
    noisy_signal = signal + noise
    return noisy_signal

def main():
    wav_path = "example.wav"
    FRAME_SIZE = 1024  # 可调节帧大小
    SNR_TEST_LEVELS = [20, 10, 5, 0]  # dB，测试不同信噪比下的抗噪性能

    # 1. 读取音频
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]  # 只用第一通道
    # 归一化
    maxv = np.max(np.abs(data))
    if maxv > 0:
        data = data / maxv

    # 2. 按帧编码/解码及时间测量（无噪声）
    n_frames = int(np.ceil(len(data) / FRAME_SIZE))
    encoded = []
    decoded = []
    total_encode_time = 0.0
    total_decode_time = 0.0
    for i in range(n_frames):
        frame = data[i*FRAME_SIZE:(i+1)*FRAME_SIZE]
        # 编码
        t0 = time.perf_counter()
        enc = adpcm9_encode(frame)
        t1 = time.perf_counter()
        # 解码
        dec = adpcm9_decode(enc)
        t2 = time.perf_counter()

        encoded.append(enc)
        decoded.append(dec)
        total_encode_time += (t1 - t0)
        total_decode_time += (t2 - t1)

    encoded = np.concatenate(encoded)
    decoded = np.concatenate(decoded)
    decoded = decoded[:len(data)]  # 对齐长度

    # 3. 计算SNR（无噪声）
    snr = compute_snr(data, decoded)
    print(f"无噪声 SNR: {snr:.2f} dB")

    # 4. 计算压缩率
    orig_size = os.path.getsize(wav_path)
    encoded_size = len(encoded) * 9 / 8  # 字节
    print(f"原始文件大小: {orig_size / 1024:.2f} KB")
    print(f"9ADPCM编码后大小: {encoded_size / 1024:.2f} KB")
    compression_ratio = orig_size / encoded_size
    print(f"压缩率: {compression_ratio:.2f}")

    # 5. 保存重构wav
    recon_wav = "adpcm9_temp.wav"
    sf.write(recon_wav, decoded, sr)
    print(f"已输出重构wav: {recon_wav}")

    # 6. 帧处理平均时间
    print(f"帧数: {n_frames}")
    print(f"平均单帧编码时间: {total_encode_time / n_frames * 1000:.3f} ms")
    print(f"平均单帧解码时间: {total_decode_time / n_frames * 1000:.3f} ms")

    # 7. 抗噪性能测试
    print("\n--- 抗噪性能测试 ---")
    for snr_db in SNR_TEST_LEVELS:
        print(f"\n添加背景噪声: SNR={snr_db} dB")
        noisy_data = add_noise(data, snr_db)
        encoded_noisy = []
        decoded_noisy = []
        for i in range(n_frames):
            frame = noisy_data[i*FRAME_SIZE:(i+1)*FRAME_SIZE]
            enc = adpcm9_encode(frame)
            dec = adpcm9_decode(enc)
            encoded_noisy.append(enc)
            decoded_noisy.append(dec)
        encoded_noisy = np.concatenate(encoded_noisy)
        decoded_noisy = np.concatenate(decoded_noisy)[:len(data)]

        # SNR1: 原始-noisy的SNR
        snr_input = compute_snr(data, noisy_data)
        # SNR2: 原始-经ADPCM后重构的SNR
        snr_output = compute_snr(data, decoded_noisy)
        print(f"输入信号与原始信号SNR: {snr_input:.2f} dB")
        print(f"ADPCM9重构信号与原始信号SNR: {snr_output:.2f} dB")
        # 也可保存重构有噪声的wav
        recon_noisy_wav = f"adpcm9_temp_noise_{snr_db}dB.wav"
        sf.write(recon_noisy_wav, decoded_noisy, sr)
        print(f"已输出有噪声重构wav: {recon_noisy_wav}")

if __name__ == "__main__":
    main()