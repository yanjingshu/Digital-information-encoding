import numpy as np
import soundfile as sf
from scipy.signal import lfilter
import os

def lpc(signal, order):
    from scipy.linalg import toeplitz, solve
    R = np.correlate(signal, signal, mode='full')
    R = R[len(signal)-1:len(signal)+order]
    r = R[1:]
    Rmat = toeplitz(R[:-1])
    # 防止矩阵奇异
    Rmat += np.eye(order) * 1e-6
    a = solve(Rmat, -r)
    return np.concatenate(([1.], a))

def celp_encode(signal, order=10):
    frame_len = 160  # 20ms @ 8kHz
    frames = [
        signal[i:i+frame_len]
        for i in range(0, len(signal), frame_len)
    ]
    encoded = []
    for frame in frames:
        if len(frame) < order + 1 or np.allclose(frame, 0):
            encoded.extend(frame)
            continue
        a = lpc(frame, order)
        pred = lfilter([0] + -a[1:], [1], frame)
        resi = frame - pred
        resi_q = np.round(resi * 128) / 128
        recon = lfilter([1], a, resi_q)
        encoded.extend(recon)
    return np.array(encoded, dtype=np.float32)

def compute_snr(original, reconstructed):
    minlen = min(len(original), len(reconstructed))
    noise = original[:minlen] - reconstructed[:minlen]
    snr = 10 * np.log10(np.sum(original[:minlen] ** 2) / np.sum(noise ** 2))
    return snr

def main():
    wav_path = "example.wav"
    out_wav = "celp.wav"
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]  # 只处理单通道
    maxv = np.abs(data).max()
    if maxv > 0:
        data = data / maxv

    recon = celp_encode(data, order=10)

    snr = compute_snr(data, recon)
    print(f"SNR: {snr:.2f} dB")

    orig_size = os.path.getsize(wav_path)
    # 8bit残差+LPC参数(忽略不计)，估算压缩后每采样1字节
    celp_size = len(recon)  # 每采样1字节（假设8bit量化）
    print(f"原始WAV大小: {orig_size/1024:.2f} KB")
    print(f"CELP编码后估算大小: {celp_size/1024:.2f} KB")
    compression_ratio = orig_size / celp_size
    print(f"压缩率: {compression_ratio:.2f}")

    sf.write(out_wav, recon, sr)
    print(f"输出: {out_wav}")

if __name__ == "__main__":
    main()