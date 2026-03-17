import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from air_track.detector.utils.detect_utils import is_all_nan


def draw_feature_img_orig_data(feature_img, save_path, is_plt_star=False, max_iloc_x=0, max_iloc_y=0):
    """将标签特征图或者模型输出的特征图可视化，使用原始数据直接保存"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 将归一化后的矩阵直接保存
    if len(feature_img.shape) == 2:
        plt.figure()
        plt.imshow(feature_img)
        if is_plt_star:
            plt.plot(max_iloc_x, max_iloc_y, 'r*')
        plt.colorbar()
        plt.savefig(save_path)
        plt.close()
    else:
        for i, item in enumerate(feature_img):
            plt.figure()
            plt.imshow(feature_img[i])
            if is_plt_star:
                plt.plot(max_iloc_x, max_iloc_y, 'r*')
            plt.colorbar()
            plt.savefig(save_path.split('.')[0] + f'_channel_{i}.png')
            plt.close()


def draw_feature_img(feature_img, save_path, max_pixel=255, use_gray=False):
    """将标签特征图或者模型输出的特征图可视化"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 将归一化后的矩阵转换为灰度值（0-255范围内）
    if len(feature_img.shape) == 2:
        img = (feature_img * max_pixel)
    else:
        if use_gray:
            img = (feature_img * max_pixel)[0]
        else:
            img = (feature_img * max_pixel).transpose(1, 2, 0)[:, :, -3:]

    if max_pixel == np.iinfo(np.uint16).max:
        img = img.astype(np.uint16)
    else:
        img = img.astype(np.uint8)

    img.clip(0, max_pixel)

    # 保存图片
    cv2.imwrite(save_path, img)


def show_distance(x_label, y_label, x_pred, y_pred, save_path, file_name='distance.png'):
    """可视化distance并保存在指定文件夹下"""
    os.makedirs(save_path, exist_ok=True)

    y_label = np.array(y_label) / 1e3
    y_pred = np.array(y_pred) / 1e3

    rmse_value = 0
    if len(y_pred) != 0:
        for i in range(len(x_pred)):
            rmse_value += (y_pred[i] - y_label[i]) ** 2
        if rmse_value != 0:
            rmse_value = np.sqrt(rmse_value / len(x_pred))
    else:
        rmse_value = np.nan

    save_file = os.path.join(save_path, file_name)
    base_name = file_name.split('.')[0][:-9]
    # 绘制折线图
    plt.figure()

    # 绘制y_axis_1的数据点，颜色设置为红色
    plt.scatter(x_label, y_label, s=1, c='red', label='Ground Truth')  # s参数控制点的大小
    # 绘制y_axis_2的数据点，颜色设置为蓝色
    if len(y_pred) != 0:
        plt.scatter(x_pred, y_pred, s=1, c='blue', label='Estimation')  # s参数控制点的大小

    plt.title(f'{base_name} RMSE: {rmse_value}')
    plt.legend()
    plt.xlabel('Frame ID')
    plt.ylabel('Distance (Km)')

    plt.savefig(save_file)

    plt.close()


def show_scatter_plot(x, y, save_path, title='show', xlabel='Frame ID', ylabel='y', label='', color='red'):
    """可视化snr并保存在指定文件夹下"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x_show, y_show = [], []
    for i, j in zip(x, y):
        if not np.isnan(j):
            x_show.append(i)
            y_show.append(j)

    # 绘制折线图
    plt.figure()

    # 绘制y_axis_1的数据点，颜色设置为红色
    plt.scatter(x, y, s=1, c=color, label=label)  # s参数控制点的大小

    plt.title(title)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid('on')

    plt.savefig(save_path)

    plt.close()


def show_confidence(pos_conf_list, neg_conf_list, save_path, file_name='conf.png'):
    """可视化confidence并保存在指定文件夹下"""
    os.makedirs(save_path, exist_ok=True)

    # 绘制散点图
    plt.figure(figsize=(10, 6))

    # 绘制正例的散点
    for i, pos_confidences in enumerate(pos_conf_list):
        x = [i] * len(pos_confidences)
        plt.scatter(x, pos_confidences, color='green', s=2)

    # 绘制负例的散点
    for j, neg_confidences in enumerate(neg_conf_list):
        x = [j] * len(neg_confidences)
        plt.scatter(x, neg_confidences, color='red', s=2)

    plt.scatter([], [], color='green', label='Positive')
    plt.scatter([], [], color='red', label='Negative')

    # 添加图例
    plt.legend()

    # 设置标题和轴标签
    plt.title('Confidence Visualization')
    plt.xlabel('Index')
    plt.ylabel('Confidence')

    # 保存图像
    plt.savefig(os.path.join(save_path, file_name))

    plt.close()


def show_confidence_sorted(pos_conf_list, neg_conf_list, save_path, file_name='conf.png'):
    """可视化排序后的confidence并保存在指定文件夹下"""
    os.makedirs(save_path, exist_ok=True)

    # 清理空列表，并提取confidence值
    pos_conf_values = [conf for sublist in pos_conf_list for conf in sublist if conf]
    neg_conf_values = [conf for sublist in neg_conf_list for conf in sublist if conf]
    pos_conf_values.sort()
    neg_conf_values.sort()

    # 绘制散点图
    plt.figure(figsize=(10, 6))

    # 绘制正例的散点
    plt.scatter(range(len(pos_conf_values)), pos_conf_values, color='green', label='Positive', s=2)

    # 绘制负例的散点
    plt.scatter(range(len(neg_conf_values)), neg_conf_values, color='red', label='Negative', s=2)

    # 添加图例
    plt.legend()

    # 设置标题和轴标签
    plt.title('Confidence Visualization')
    plt.xlabel('Index')
    plt.ylabel('Confidence')

    # 保存图像
    plt.savefig(os.path.join(save_path, file_name))

    plt.close()


def show_single_scatter_plot(x, y, save_path, file_name='img.png', title='show', xlabel='Frame ID', ylabel='y',
                             label='', color='red', s=2, sort=False):
    """可视化散点图并保存在指定文件夹下"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if sort:
        y = sorted(y)

    # 绘制折线图
    plt.figure()

    # 绘制散点
    plt.scatter(x, y, s=s, c=color, label=label)  # s参数控制点的大小

    plt.title(title)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid('on')

    plt.savefig(os.path.join(save_path, file_name))
    plt.close()


def show_bar_chart(data, save_path, file_name='img.png', range_min=0, range_max=1, range_interval=0.005,
                   title='Bar Chart', xlabel='Confidence', ylabel='Count/Scale'):
    """可视化柱状图并保存"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 确保数据列表非空
    if not data:
        print('show_bar_chart的数据列表为空')
        return

    # 计算总数
    total = len(data)

    # 生成x坐标，从0到1，以0.005为间隔
    x = np.arange(range_min, range_max, range_interval)
    # 计算每个区间的数量
    counts = []
    for i in range(len(x) - 1):
        if i == len(x) - 2:
            count = sum(1 for value in data if x[i] <= value <= x[i + 1])
        else:
            count = sum(1 for value in data if x[i] <= value < x[i + 1])
        counts.append(count)

    # 计算比例
    proportions = [count / total for count in counts]

    # 计算柱状图的左边缘位置（每个柱子的中心）
    left = x[:-1] + range_interval / 2

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(left, counts, width=range_interval, label='Counts', edgecolor='black', linewidth=2)

    # 设置x轴刻度和标签
    ax.set_xticks(x)
    ax.tick_params(axis='x', labelrotation=45)

    # 在每一条柱子上边写上数量/比例
    for bar, count, proportion in zip(bars, counts, proportions):
        height = bar.get_height()
        ax.annotate(f'{count}\n{proportion:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    # 设置图表标题和坐标轴标签
    plt.title(title + f'  Sample Num: {total}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 保存图表
    plt.savefig(os.path.join(save_path, file_name))
    plt.close()


def visualize_and_save(img_path, detected_objects=None, visualize_save_dir='output',
                       scale_w=1, scale_h=1, bbox_gts=None, distance=None):
    """检测结果可视化"""
    image = cv2.imread(img_path)
    # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    save_file_name = os.path.basename(img_path)
    os.makedirs(visualize_save_dir, exist_ok=True)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if bbox_gts and not is_all_nan(bbox_gts):
        for bbox_gt in bbox_gts:
            cls_id = int(bbox_gt[0])
            bbox_gt = bbox_gt[-4:]  # -5即0位置上是类别
            if not is_all_nan(bbox_gt):
                cv2.rectangle(image, (int(bbox_gt[0]), int(bbox_gt[1])),
                              (int(bbox_gt[2] + 0.5), int(bbox_gt[3] + 0.5)), (0, 255, 0), 1)

    if distance:
        cv2.putText(image, f'distance: {distance}', (15, 25), font, 1, (0, 255, 0), 1)

    if detected_objects:
        for res in detected_objects:
            conf = float(res['conf'])
            if 'offset' in res:
                cx = (res['cx'] + res['offset'][0]) / scale_w
                cy = (res['cy'] + res['offset'][1]) / scale_h
            else:
                cx = res['cx'] / scale_w
                cy = res['cy'] / scale_h
            w = res['w'] / scale_w
            h = res['h'] / scale_h
            # 转换为左上角的坐标xy，右下角的坐标xy
            bbox = [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2 + 0.5), int(cy + h / 2 + 0.5)]

            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
            cv2.putText(image, f'{res["cls"]} {round(conf, 4)}', (bbox[0], bbox[1]), font, 0.5, (0, 0, 255), 1)

    cv2.imwrite(f'{visualize_save_dir}/{save_file_name}', image)


if __name__ == '__main__':
    import pickle
    from air_track.detector.utils.calculate_metrics import calculate_metrics

    cache_file = '/home/csz_changsha/PycharmProjects/pythonProject/AirTrack_aot_local/output/cache_metrics.pkl'
    save_path = '/home/csz_changsha/PycharmProjects/pythonProject/AirTrack_aot_local/output'

    # 使用 'rb' 模式打开文件进行读取（'rb' 代表 'read binary'）
    with open(cache_file, 'rb') as file:
        # 使用 pickle.load 来读取文件中的数据
        data = pickle.load(file)

    bbox_gts = data['bbox_gts']
    candidate_targets = data['candidate_targets']
    frame_nums = data['frame_nums']
    scale_w = data['scale_w']
    scale_h = data['scale_h']
    detection_rate, pos_pred_x, distance_pred_y, pos_conf_list, neg_conf_list, \
    pos_candidate_targets, neg_candidate_targets = \
        calculate_metrics(bbox_gts, candidate_targets, frame_nums, scale_w, scale_h,
                          threshold=0.25, target_min_size=10, iou=True)

    show_confidence(pos_conf_list, neg_conf_list, save_path, file_name='conf.png')

    show_confidence_sorted(pos_conf_list, neg_conf_list,
                           save_path=save_path,
                           file_name='conf_sorted.png')
