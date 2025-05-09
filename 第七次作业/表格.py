import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
channel_ber = np.linspace(0.001, 0.1, 10)
parity_decoded_ber = channel_ber * 2
hamming_decoded_ber = channel_ber ** 2
crc_decoded_ber = np.ones_like(channel_ber) * 1e-1
convolutional_decoded_ber = channel_ber ** 3

# 绘制图表
plt.figure(figsize=(10, 6))
plt.plot(channel_ber, parity_decoded_ber, marker='o', label='Parity', color='blue')
plt.plot(channel_ber, hamming_decoded_ber, marker='o', label='Hamming', color='orange')
plt.plot(channel_ber, crc_decoded_ber, marker='o', label='CRC', color='green')
plt.plot(channel_ber, convolutional_decoded_ber, marker='o', label='Convolutional', color='red')

# 设置对数坐标轴
plt.xscale('log')
plt.yscale('log')

# 标题和标签
plt.title('Error Correction Performance Comparison')
plt.xlabel('Channel Bit Error Rate (BER)')
plt.ylabel('Decoded BER')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 图例
plt.legend()

# 显示图表
plt.show()