import os
import cv2
from air_track.detector.utils.detect_utils import cxcywhn2xyxy


"""将单个标签和模型预测的bbox可视化在图像中"""


def draw_label_in_img(img, top_left, bottom_right, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if top_left and bottom_right:
        x_min, y_min = top_left
        x_max, y_max = bottom_right

        # 图像上绘制框
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)  # 在图像上绘制矩形

    # 保存带有标签框的图像
    cv2.imwrite(save_path, img)
    print('saved: ', save_path)


if __name__ == '__main__':
    path = '/home/csz_changsha/data/RGBT-Tiny/test_1/rgb/DJI_0179_2/00000.jpg'
    save_path = 'User_channel_1_000003.png'

    img = cv2.imread(path)

    label_path = '/home/csz_changsha/data/RGBT-Tiny/test_1/ir/DJI_0179_2/labels/00000.txt'

    with open(label_path, 'r') as file:
        labels = file.readlines()
    labels = [label.strip().split() for label in labels]  # 假设标签是空格分隔的
    # 将标签由字符串转换为浮点数
    labels = [[float(item) for item in label] for label in labels]

    pred_path = '/home/csz_changsha/data/RGBT-Tiny/test_1/label/DJI_0179_2/00000.txt'
    with open(pred_path, 'r') as file:
        preds = file.readlines()
    preds = [pred.strip().split() for pred in preds]  # 假设标签是空格分隔的
    # 将标签由字符串转换为浮点数
    preds = [[float(item) for item in pred] for pred in preds]

    for label in labels:
        bbox_gt = cxcywhn2xyxy(label[1:], img_w=640, img_h=512)
        top_left = (int(bbox_gt[0]), int(bbox_gt[1]))
        bottom_right = (int(bbox_gt[2]), int(bbox_gt[3]))

        if top_left and bottom_right:
            x_min, y_min = top_left
            x_max, y_max = bottom_right

            # 图像上绘制框
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)  # 在图像上绘制矩形

    for label in preds:
        bbox_gt = cxcywhn2xyxy(label[1:], img_w=640, img_h=512)
        top_left = (int(bbox_gt[0]), int(bbox_gt[1]))
        bottom_right = (int(bbox_gt[2]), int(bbox_gt[3]))

        if top_left and bottom_right:
            x_min, y_min = top_left
            x_max, y_max = bottom_right

            # 图像上绘制框
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)  # 在图像上绘制矩形

    # 保存带有标签框的图像
    cv2.imwrite(save_path, img)
    print('saved: ', save_path)

