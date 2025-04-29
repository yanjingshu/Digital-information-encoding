import numpy as np
import soundfile as sf
from scipy.signal import lfilter, hamming
import os
import time  # 新增

def lpc(signal, order):
    from scipy.linalg import toeplitz, solve
    R = np.correlate(signal, signal, mode='full')
    R = R[len(signal)-1:len(signal)+order]
    r = R[1:]
    Rmat = toeplitz(R[:-1])
    Rmat += np.eye(order) * 1e-6
    try:
        a = solve(Rmat, -r)
    except Exception:
        # fallback，直接返回全通滤波器
        return np.ones(order+1)
    return np.concatenate(([1.], a))

def celp_encode(signal, order=10, frame_len=160, overlap=80, measure_time=False):
    # 汉明窗
    win = hamming(frame_len, sym=False)
    step = frame_len - overlap
    num_frames = (len(signal) - overlap) // step

    recon = np.zeros(len(signal))
    win_sum = np.zeros(len(signal))  # 用于重叠加窗归一化

    frame_times = []  # 新增：存储每帧处理时间

    for i in range(num_frames):
        start = i * step
        end = start + frame_len
        if end > len(signal):
            break
        frame = signal[start:end] * win
        if np.allclose(frame, 0):
            continue

        t0 = time.perf_counter() if measure_time else None  # 新增

        a = lpc(frame, order)
        pred = lfilter([0] + -a[1:], [1], frame)
        resi = frame - pred
        # 不量化残差，直接合成
        recon_frame = lfilter([1], a, resi)
        # 归一化能量
        if np.max(np.abs(recon_frame)) > 1e-3:
            recon_frame *= np.max(np.abs(frame)) / (np.max(np.abs(recon_frame)) + 1e-8)
        # 重叠加窗
        recon[start:end] += recon_frame * win
        win_sum[start:end] += win**2

        if measure_time:
            t1 = time.perf_counter()
            frame_times.append(t1 - t0)

    # 避免重叠区域能量过高
    win_sum[win_sum == 0] = 1e-8
    recon /= win_sum
    recon = np.clip(recon, -1, 1)
    if measure_time:
        return recon.astype(np.float32), frame_times
    else:
        return recon.astype(np.float32)

def compute_snr(original, reconstructed):
    minlen = min(len(original), len(reconstructed))
    noise = original[:minlen] - reconstructed[:minlen]
    snr = 10 * np.log10(np.sum(original[:minlen] ** 2) / (np.sum(noise ** 2) + 1e-10))
    return snr

def main():
    wav_path = "example.wav"
    out_wav = "celp_improved.wav"
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    maxv = np.abs(data).max()
    if maxv > 0:
        data = data / maxv

    recon, frame_times = celp_encode(data, order=10, frame_len=160, overlap=80, measure_time=True)

    snr = compute_snr(data, recon)
    print(f"SNR: {snr:.2f} dB")

    orig_size = os.path.getsize(wav_path)
    celp_size = len(recon)
    compression_ratio = orig_size / celp_size
    print(f"原始WAV大小: {orig_size/1024:.2f} KB")
    print(f"CELP编码后估算大小: {celp_size/1024:.2f} KB")
    print(f"压缩率: {compression_ratio:.2f}")

    # 输出帧处理时间统计
    frame_times = np.array(frame_times)
    print(f"单帧平均处理时间: {frame_times.mean()*1000:.3f} ms")
    print(f"单帧最大处理时间: {frame_times.max()*1000:.3f} ms")
    print(f"单帧最小处理时间: {frame_times.min()*1000:.3f} ms")

    sf.write(out_wav, recon, sr)
    print(f"输出: {out_wav}")

if __name__ == "__main__":
    main()