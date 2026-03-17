import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from skimage import io, img_as_float


def process_gray(img_pth, lp_cut_off=None, hp_cut_off=None, if_savefig=False, save_img_pth="output"):
    # 读取原始灰度图像（用于后续保存时获取原始类型和动态范围）
    orig_gray = io.imread(img_pth, as_gray=True)
    orig_dtype = orig_gray.dtype

    # 为了傅里叶处理，我们转换为 float 类型
    gray_image = orig_gray.astype(float)

    # 对图像进行傅里叶变换并移位
    f_transform = fftpack.fft2(gray_image)
    f_transform_shifted = fftpack.fftshift(f_transform)

    # 计算原始频域的对数幅值（用于显示）
    orig_mag = np.log(np.abs(f_transform_shifted) + 1)

    # 定义带通滤波函数（支持低通、高通或带通）
    def bandpass_filter(f_transform_shifted, lp_cut_off, hp_cut_off):
        rows, cols = f_transform_shifted.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), dtype=float)
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                if lp_cut_off is not None and hp_cut_off is None:
                    # 仅 lp_cut_off：低通滤波
                    if distance <= lp_cut_off:
                        mask[i, j] = 1
                elif hp_cut_off is not None and lp_cut_off is None:
                    # 仅 hp_cut_off：高通滤波
                    if distance >= hp_cut_off:
                        mask[i, j] = 1
                elif hp_cut_off is not None and lp_cut_off is not None:
                    # 同时提供：带通滤波，保留频率在 hp_cut_off 和 lp_cut_off 之间的部分
                    if hp_cut_off <= distance <= lp_cut_off:
                        mask[i, j] = 1
                else:
                    mask[i, j] = 1
        return f_transform_shifted * mask

    # 应用带通滤波
    f_transform_shifted_filtered = bandpass_filter(f_transform_shifted, lp_cut_off, hp_cut_off)

    # 计算滤波后频域的对数幅值（用于显示）
    filtered_mag = np.log(np.abs(f_transform_shifted_filtered) + 1)

    # 逆傅里叶变换得到滤波后的空间域图像
    f_transform_filtered = fftpack.ifftshift(f_transform_shifted_filtered)
    image_filtered = np.abs(fftpack.ifft2(f_transform_filtered))

    # 绘制结果图像（显示部分子图的坐标轴情况）
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(gray_image, cmap='gray')
    axes[0, 0].set_title('Space Domain (Original Image)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(orig_mag, cmap='gray')
    axes[0, 1].set_title('Frequency Domain (Original)')
    axes[0, 1].axis('on')

    axes[0, 2].imshow(filtered_mag, cmap='gray')
    axes[0, 2].set_title('Filtered Frequency Domain')
    axes[0, 2].axis('on')

    axes[1, 0].imshow(image_filtered, cmap='gray')
    axes[1, 0].set_title('Space Domain (Filtered)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(filtered_mag, cmap='gray')
    axes[1, 1].set_title('Filtered Frequency (dup.)')
    axes[1, 1].axis('on')

    axes[1, 2].imshow(image_filtered, cmap='gray')
    axes[1, 2].set_title('Filtered Space (dup.)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    # 保存整幅图像（展示图）如果需要
    if if_savefig:
        plt.savefig(save_img_pth + "_gray.png")
    plt.show()

    # 处理后的图像保存：将 image_filtered 转换为与原始图像同样的类型
    # 假设若原始为整数类型，则动态范围为 [0, 255]（否则可根据实际情况调整）
    if np.issubdtype(orig_dtype, np.integer):
        proc_img = np.clip(image_filtered, 0, 255).astype(orig_dtype)
    else:
        proc_img = image_filtered

    # 保存处理后的图像（仅保存图像数据，不含子图、坐标轴等）
    io.imsave(save_img_pth + "_gray.png", proc_img)

    # 另外显示归一化后的处理图像（仅展示用途，不用于保存）
    def normalize_img(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    norm_image_filtered = normalize_img(image_filtered)

    plt.figure(figsize=(5, 5))
    plt.imshow(norm_image_filtered, cmap='gray')
    plt.title('Normalized Filtered Space Domain')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()

    # 返回原始和滤波后的频域矩阵（对数幅值）
    return orig_mag, filtered_mag


def process_rgb(img_pth, lp_cut_off=None, hp_cut_off=None, if_savefig=False, save_img_pth="output"):
    img_name = os.path.basename(img_pth)
    # 读取原始RGB图像，并保存原始 dtype（用于后续保存）
    orig_rgb = io.imread(img_pth)
    orig_dtype = orig_rgb.dtype  # 比如通常为 uint8
    # 为处理方便转换为浮点数
    image = img_as_float(orig_rgb)

    # 分离RGB通道
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]

    # 定义傅里叶变换及移位函数
    def fourier_transform(channel):
        f_transform = fftpack.fft2(channel)
        return fftpack.fftshift(f_transform)

    # 定义带通滤波函数（同 process_gray 中）
    def bandpass_filter(f_transform_shifted, lp_cut_off, hp_cut_off):
        rows, cols = f_transform_shifted.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols), dtype=float)
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                if lp_cut_off is not None and hp_cut_off is None:
                    if distance <= lp_cut_off:
                        mask[i, j] = 1
                elif hp_cut_off is not None and lp_cut_off is None:
                    if distance >= hp_cut_off:
                        mask[i, j] = 1
                elif hp_cut_off is not None and lp_cut_off is not None:
                    if hp_cut_off <= distance <= lp_cut_off:
                        mask[i, j] = 1
                else:
                    mask[i, j] = 1
        return f_transform_shifted * mask

    # 针对单通道的处理
    def process_channel(channel, lp_cut_off, hp_cut_off):
        f_transform_shifted = fourier_transform(channel)
        orig_log_mag = np.log(np.abs(f_transform_shifted) + 1)
        f_transform_shifted_filtered = bandpass_filter(f_transform_shifted, lp_cut_off, hp_cut_off)
        filtered_log_mag = np.log(np.abs(f_transform_shifted_filtered) + 1)
        # 逆变换回空间域
        f_transform_filtered = fftpack.ifftshift(f_transform_shifted_filtered)
        channel_filtered = np.abs(fftpack.ifft2(f_transform_filtered))
        return orig_log_mag, filtered_log_mag, channel_filtered

    # 分别处理三个通道
    r_orig, r_filtered, r_filtered_img = process_channel(r_channel, lp_cut_off, hp_cut_off)
    g_orig, g_filtered, g_filtered_img = process_channel(g_channel, lp_cut_off, hp_cut_off)
    b_orig, b_filtered, b_filtered_img = process_channel(b_channel, lp_cut_off, hp_cut_off)

    # 合并滤波后的空间域图像
    image_filtered = np.stack([r_filtered_img, g_filtered_img, b_filtered_img], axis=-1)

    # 绘制结果图像
    # fig, axes = plt.subplots(2, 3, figsize=(15, 12))

    # axes[0, 0].imshow(image)
    # axes[0, 0].set_title('RGB Image (Original)')
    # axes[0, 0].axis('off')
    #
    # axes[0, 1].imshow(r_orig, cmap='gray')
    # axes[0, 1].set_title('R Channel Frequency (Orig)')
    # axes[0, 1].axis('on')
    #
    # axes[0, 2].imshow(r_filtered, cmap='gray')
    # axes[0, 2].set_title('R Channel Frequency (Filt.)')
    # axes[0, 2].axis('on')
    #
    # axes[1, 0].imshow(image_filtered)
    # axes[1, 0].set_title('RGB Image (Filtered)')
    # axes[1, 0].axis('off')
    #
    # axes[1, 1].imshow(g_orig, cmap='gray')
    # axes[1, 1].set_title('G Channel Frequency (Orig)')
    # axes[1, 1].axis('on')
    #
    # axes[1, 2].imshow(g_filtered, cmap='gray')
    # axes[1, 2].set_title('G Channel Frequency (Filt.)')
    # axes[1, 2].axis('on')

    # plt.tight_layout()
    # if if_savefig:
    #     plt.savefig(os.path.join(save_img_pth, img_name))
    # plt.show()

    # 将处理后的RGB图像转换为与原始图像相同类型和动态范围
    if np.issubdtype(orig_dtype, np.integer):
        # 假设原始 RGB 图像的整数类型通常为 0-255
        proc_rgb = np.clip(image_filtered * 255, 0, 255).astype(orig_dtype)
    else:
        proc_rgb = image_filtered

    # 保存处理后的 RGB 图像（保存为与原始同尺寸、同类型）
    io.imsave(os.path.join(save_img_pth, img_name), proc_rgb)

    # 对滤波后图像进行归一化显示（仅用于展示）
    def normalize_img(img):
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    norm_image_filtered = normalize_img(image_filtered)

    # plt.figure(figsize=(5, 5))
    # plt.imshow(norm_image_filtered)
    # plt.title('Normalized Filtered RGB Image')
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.tight_layout()
    # plt.show()

    # 返回每个通道的频域对数幅值矩阵
    return {
        "r_channel_freq": r_orig,
        "g_channel_freq": g_orig,
        "b_channel_freq": b_orig,
        "r_filtered_freq": r_filtered,
        "g_filtered_freq": g_filtered,
        "b_filtered_freq": b_filtered
    }


if __name__ == '__main__':
    path = '/home/csz_changsha/data/yanqing_data/0326dabaige/train/images'
    save_path = '/home/csz_changsha/data/yanqing_data/0326dabaige/train/images_filtered'
    os.makedirs(save_path, exist_ok=True)

    img_paths = glob.glob(os.path.join(path, '*.jpg'))
    img_paths.sort()

    for i, img_pth in enumerate(img_paths):
        print(f'当前处理{i}/{len(img_paths)}, img_path={img_pth}')
        result_gray = process_rgb(img_pth, lp_cut_off=None, hp_cut_off=1, if_savefig=False, save_img_pth=save_path)
