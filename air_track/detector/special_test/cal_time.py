import numpy as np


def calculate_trimmed_mean(file_path):
    # 读取文件中的所有耗时数据
    with open(file_path, 'r') as f:
        times = [float(line.strip()) for line in f if line.strip()]

    if len(times) < 3:
        print(f"数据点不足（只有 {len(times)} 个），无法去掉最大最小值")
        return np.mean(times) if times else 0

    # 转换为numpy数组
    times_array = np.array(times)

    # 去掉最大值和最小值
    trimmed_times = np.delete(times_array, [np.argmax(times_array), np.argmin(times_array)])

    # 计算均值
    mean_time = np.mean(trimmed_times)

    print(f"原始数据: {times}")
    print(f"处理后数据: {trimmed_times}")
    print(f"去掉最大最小值后的平均耗时: {mean_time:.3f} ms")
    print(f"处理掉的数据点: 最大值 {np.max(times_array):.3f} ms, 最小值 {np.min(times_array):.3f} ms")

    return mean_time


# 使用示例
calculate_trimmed_mean("/home/linana/inference_times.txt")