import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

# ======================
# 参数设置
# ======================
TEST_BITS = 10000  # 每次测试的比特数
ERROR_RATES = np.logspace(-3, -1, 10)  # 测试误码率范围：0.1%~10%
CODING_SCHEMES = ['Parity', 'Hamming', 'CRC', 'Convolutional']
CHANNEL_TYPES = ['BSC', 'BEC', 'AWGN', 'Burst']

# ======================
# 辅助函数
# ======================
def generate_random_bits(n_bits):
    """ 生成随机比特序列 """
    return np.random.randint(0, 2, n_bits).tolist()

def simulate_bsc_channel(bits, error_prob):
    """ 模拟二进制对称信道 """
    errors = np.random.rand(len(bits)) < error_prob
    return [bit ^ 1 if err else bit for bit, err in zip(bits, errors)]

def simulate_bec_channel(bits, erasure_prob):
    """ 模拟二进制非对称信道（擦除信道） """
    erasures = np.random.rand(len(bits)) < erasure_prob
    return [None if err else bit for bit, err in zip(bits, erasures)]

def simulate_awgn_channel(bits, snr_db):
    """ 模拟高斯白噪声信道 """
    bits = np.array(bits) * 2 - 1  # BPSK 调制
    snr = 10 ** (snr_db / 10)
    noise_power = 1 / snr
    noise = np.random.normal(0, np.sqrt(noise_power), len(bits))
    received = bits + noise
    decoded = (received > 0).astype(int)
    return decoded.tolist()

def simulate_burst_channel(bits, burst_prob, burst_length):
    """ 模拟突发错误信道 """
    received = bits.copy()
    i = 0
    while i < len(bits):
        if np.random.rand() < burst_prob:
            burst_end = min(i + burst_length, len(bits))
            for j in range(i, burst_end):
                received[j] ^= 1
            i = burst_end
        else:
            i += 1
    return received

# ======================
# 编码/解码函数（需提前实现）
def parity_encode(data):
    """ 奇偶校验编码（返回编码后的数据） """
    parity = sum(data) % 2
    return data + [parity]

def parity_decode(received):
    """ 奇偶校验解码（仅检错，不纠错） """
    if None in received:  # 处理擦除信道
        return None
    return received[:-1] if sum(received) % 2 == 0 else None

# 7-4汉明码
def hamming_encode(data_4bits):
    """
    汉明码(7,4)编码
    :param data_4bits: 4位数据（列表形式）
    :return: 7位编码（p1,p2,d1,p3,d2,d3,d4）
    """
    p1 = data_4bits[0] ^ data_4bits[1] ^ data_4bits[3]
    p2 = data_4bits[0] ^ data_4bits[2] ^ data_4bits[3]
    p3 = data_4bits[1] ^ data_4bits[2] ^ data_4bits[3]
    return [p1, p2, data_4bits[0], p3, data_4bits[1], data_4bits[2], data_4bits[3]]

def hamming_decode(received_7bits):
    """
    汉明码纠错解码
    :return: 校正后的4位数据
    """
    if None in received_7bits:  # 处理擦除信道
        return None
    # 校验子计算
    s1 = received_7bits[0] ^ received_7bits[2] ^ received_7bits[4] ^ received_7bits[6]
    s2 = received_7bits[1] ^ received_7bits[2] ^ received_7bits[5] ^ received_7bits[6]
    s3 = received_7bits[3] ^ received_7bits[4] ^ received_7bits[5] ^ received_7bits[6]
    error_pos = s1 + 2 * s2 + 4 * s3 - 1  # 错误位置索引

    # 纠正错误
    if error_pos >= 0:
        received_7bits[error_pos] ^= 1

    return [received_7bits[2], received_7bits[4], received_7bits[5], received_7bits[6]]

def crc_encode(data_bytes, poly=0x07):
    """
    CRC-8编码（生成多项式：x^8 + x^2 + x + 1）
    :param data_bytes: 输入字节数组
    :param poly: 生成多项式（默认0x07）
    :return: 带CRC校验码的字节流
    """
    crc = 0
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFF
    return data_bytes + bytes([crc])

# CRC-8编解码
def crc_check(encoded_bytes, poly=0x07):
    """
    CRC校验，返回是否通过
    """
    crc = 0
    for byte in encoded_bytes:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ poly
            else:
                crc <<= 1
            crc &= 0xFF
    return crc == 0

# 卷积码
class ConvolutionalEncoder:
    """ (3,1,3)卷积码编码器，生成多项式：G1=0o7, G2=0o5, G3=0o3 """

    def __init__(self):
        self.state = [0, 0, 0]  # 移位寄存器状态

    def encode_bit(self, bit):
        self.state = [bit] + self.state[:-1]
        # 计算输出位
        out1 = bit ^ self.state[1] ^ self.state[2]
        out2 = bit ^ self.state[0] ^ self.state[2]
        out3 = bit ^ self.state[0] ^ self.state[1]
        return [out1, out2, out3]

    def encode(self, bits):
        encoded = []
        for b in bits:
            encoded += self.encode_bit(b)
        # 清空移位寄存器（尾部补零）
        for _ in range(3):
            encoded += self.encode_bit(0)
        return encoded

# ======================
# BER测试主逻辑
# ======================
def calculate_ber(encoder_func, decoder_func, channel_func, channel_param, n_tests=100):
    """
    计算指定编码方案的BER
    :param encoder_func: 编码函数（输入原始bits，返回编码后bits）
    :param decoder_func: 解码函数（输入接收bits，返回解码bits或None）
    :param channel_func: 信道模拟函数
    :param channel_param: 信道参数
    :param n_tests: 测试次数（取平均）
    """
    total_errors = 0
    total_bits = 0

    for _ in range(n_tests):
        # 生成随机数据
        original = generate_random_bits(TEST_BITS)

        # 编码
        encoded = encoder_func(original)

        # 通过噪声信道
        received = channel_func(encoded, channel_param)

        # 解码
        decoded = decoder_func(received)

        # 统计错误
        if decoded is None:  # 检错失败（如CRC校验不通过）
            total_errors += len(original)  # 假设全部丢弃，所有bit均错误
        else:
            errors = sum(a != b for a, b in zip(original, decoded))
            total_errors += errors

        total_bits += len(original)

    return total_errors / total_bits

# ======================
# 计算编码效率
# ======================
def calculate_coding_efficiency(encoder_func):
    """ 计算编码效率 """
    original = generate_random_bits(TEST_BITS)
    encoded = encoder_func(original)
    return len(original) / len(encoded)

# ======================
# 主测试流程
# ======================
if __name__ == "__main__":
    # 初始化结果存储
    ber_results = {channel: {scheme: [] for scheme in CODING_SCHEMES} for channel in CHANNEL_TYPES}
    coding_efficiencies = {}

    # 遍历不同信道类型
    for channel_type in CHANNEL_TYPES:
        print(f"测试信道类型：{channel_type}")
        if channel_type == 'BSC':
            channel_params = ERROR_RATES
            channel_func = simulate_bsc_channel
        elif channel_type == 'BEC':
            channel_params = ERROR_RATES
            channel_func = simulate_bec_channel
        elif channel_type == 'AWGN':
            # 将误码率转换为 SNR
            channel_params = -10 * np.log10(0.5 * erfc(np.sqrt(10 ** (np.log10(ERROR_RATES)))))
            channel_func = simulate_awgn_channel
        elif channel_type == 'Burst':
            channel_params = ERROR_RATES
            burst_length = 10  # 突发错误长度
            channel_func = lambda bits, prob: simulate_burst_channel(bits, prob, burst_length)

        for channel_param in channel_params:
            print(f"  测试信道参数：{channel_param:.4f}")

            # 测试各编码方案
            # 1. 奇偶校验
            ber = calculate_ber(
                encoder_func=parity_encode,
                decoder_func=lambda x: parity_decode(x) or x[:-1],  # 强制通过
                channel_func=channel_func,
                channel_param=channel_param
            )
            ber_results[channel_type]['Parity'].append(ber)

            # 2. 汉明码(7,4)
            def hamming_encoder(data):
                # 分块编码（每4bit转为7bit）
                encoded = []
                for i in range(0, len(data), 4):
                    chunk = data[i:i + 4]
                    if len(chunk) < 4:
                        chunk += [0] * (4 - len(chunk))  # 补零
                    encoded += hamming_encode(chunk)
                return encoded

            def hamming_decoder(received):
                decoded = []
                for i in range(0, len(received), 7):
                    chunk = received[i:i + 7]
                    if len(chunk) < 7 or None in chunk:
                        break
                    decoded += hamming_decode(chunk)
                return decoded[:TEST_BITS]  # 截断到原始长度

            ber = calculate_ber(hamming_encoder, hamming_decoder, channel_func, channel_param)
            ber_results[channel_type]['Hamming'].append(ber)

            # 3. CRC-8（假设已实现crc_encode和crc_check）
            def crc_encoder(data):
                byte_data = bytes(np.packbits(data).tolist())
                encoded_bytes = crc_encode(byte_data)
                return np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8)).tolist()

            def crc_decoder(received):
                if None in received:  # 处理擦除信道
                    return None
                byte_stream = bytes(np.packbits(received).tolist())
                if crc_check(byte_stream):
                    data_bytes = byte_stream[:-1]  # 去除CRC字节
                    return np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8)).tolist()[:TEST_BITS]
                else:
                    return None  # 检错失败

            ber = calculate_ber(crc_encoder, crc_decoder, channel_func, channel_param)
            ber_results[channel_type]['CRC'].append(ber)

            # 4. 卷积码（假设已实现ConvolutionalEncoder类）
            encoder = ConvolutionalEncoder()

            def conv_encoder(data):
                return encoder.encode(data)

            # 需实现维特比解码（此处简化示例）
            def conv_decoder(received):
                if None in received:  # 处理擦除信道
                    return None
                # 简化：直接取每3位中的第一位（仅示例）
                return [received[i] for i in range(0, len(received), 3)][:TEST_BITS]

            ber = calculate_ber(conv_encoder, conv_decoder, channel_func, channel_param)
            ber_results[channel_type]['Convolutional'].append(ber)

    # 计算编码效率
    def hamming_encoder(data):
        encoded = []
        for i in range(0, len(data), 4):
            chunk = data[i:i + 4]
            if len(chunk) < 4:
                chunk += [0] * (4 - len(chunk))  # 补零
            encoded += hamming_encode(chunk)
        return encoded

    def crc_encoder(data):
        byte_data = bytes(np.packbits(data).tolist())
        encoded_bytes = crc_encode(byte_data)
        return np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8)).tolist()

    encoder = ConvolutionalEncoder()
    def conv_encoder(data):
        return encoder.encode(data)

    coding_efficiencies['Parity'] = calculate_coding_efficiency(parity_encode)
    coding_efficiencies['Hamming'] = calculate_coding_efficiency(hamming_encoder)
    coding_efficiencies['CRC'] = calculate_coding_efficiency(crc_encoder)
    coding_efficiencies['Convolutional'] = calculate_coding_efficiency(conv_encoder)

    # ======================
    # 结果可视化
    # ======================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, channel_type in enumerate(CHANNEL_TYPES):
        ax = axes[i]
        for scheme in CODING_SCHEMES:
            if channel_type == 'AWGN':
                x_values = ERROR_RATES
            else:
                x_values = ERROR_RATES
            ax.semilogy(x_values, ber_results[channel_type][scheme], 'o-', label=scheme)

        ax.set_xlabel('Channel Parameter')
        ax.set_ylabel('Decoded BER')
        ax.set_title(f'{channel_type} Channel')
        ax.grid(True, which="both", ls="--")
        ax.legend()

    plt.tight_layout()
    plt.show()