import os
import cv2


"""将单个标签的bbox可视化在图像中"""


def draw_label_in_img(img, top_left, bottom_right, save_path, max_pixel=255):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if top_left and bottom_right:
        x_min, y_min = top_left
        x_max, y_max = bottom_right

        # 图像上绘制框
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, max_pixel, 0), 1)  # 在图像上绘制矩形

    # 保存带有标签框的图像
    cv2.imwrite(save_path, img)
    print('saved: ', save_path)


if __name__ == '__main__':
    path = '/home/csz_changsha/data/AR_2410/normal/lz_trj_1_0300/Images/lz_trj_1_0300_result/User_channel_1_000003.png'
    save_path = 'output/User_channel_1_000003.png'

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    center_xy = (411, 169)
    w = 10
    h = 10
    top_left = (int(center_xy[0] - w / 2), int(center_xy[1] - h / 2))
    bottom_right = (int(center_xy[0] + w / 2), int(center_xy[1] + h / 2))

    draw_label_in_img(img, top_left, bottom_right, save_path, max_pixel=65535)
