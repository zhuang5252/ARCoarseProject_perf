import os
import glob
import imageio


def single_folder_create_gif(image_folder, output_gif_path="output.gif", img_end='.png'):
    # 获取所有图片文件并按名称排序
    images = sorted(glob.glob(os.path.join(image_folder, f'*{img_end}')))
    # img_files = sorted(glob.glob("%s/*.bmp" % img_path))
    # images.sort(key=lambda x: int(x[len(images) + 43:-4]))
    images.sort()

    # 确定输出GIF的路径
    # output_gif_path = os.path.join(os.path.dirname(image_folder), output_gif_name)

    # 读取图片并创建gif
    with imageio.get_writer(output_gif_path, mode='I', duration=0.002) as writer:
        for i, filename in enumerate(images):
            if i % 3 == 0:
                image = imageio.imread(filename)
                # image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                writer.append_data(image)


if __name__ == '__main__':
    img_path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/123/AirTrack_results/模型测试可视化/2_and_8km/visual_result_2024_12_05_09_10×10_all_3×3_1/sah_trj_2_0730'
    output_path = os.path.join(os.path.dirname(img_path), os.path.basename(img_path)) + '.gif'

    single_folder_create_gif(img_path, output_path)
