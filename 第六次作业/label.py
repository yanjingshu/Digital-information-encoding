import matplotlib.pyplot as plt
import numpy as np

bitrates = ["MP3-VBR2", "MP3-VBR4", "MP3-VBR9", "AAC_LC128k", "AAC_LC192k"]
x = np.arange(len(bitrates))

# 数据（按你的表格填入）
cr_sample1 = [20.65, 24.28, 39.53, 11.93, 9.21]
snr_sample1 = [3.78, 3.78, 3.73, 3.78, 3.78]

cr_sample5 = [15.25, 18.98, 36.45, 11.69, 8.16]
snr_sample5 = [3.31, 3.31, 3.24, 3.32, 3.32]

width = 0.35  # 柱宽

plt.figure(figsize=(12, 5))

# 压缩率对比
plt.subplot(1, 2, 1)
plt.bar(x - width/2, cr_sample1, width, label='sample1')
plt.bar(x + width/2, cr_sample5, width, label='sample5')
plt.xticks(x, bitrates, rotation=20)
plt.xlabel('Codec Setting')
plt.ylabel('Compression Ratio')
plt.title('Compression Ratio Comparison')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# SNR对比
plt.subplot(1, 2, 2)
plt.bar(x - width/2, snr_sample1, width, label='sample1')
plt.bar(x + width/2, snr_sample5, width, label='sample5')
plt.xticks(x, bitrates, rotation=20)
plt.xlabel('Codec Setting')
plt.ylabel('SNR (dB)')
plt.title('SNR Comparison')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()