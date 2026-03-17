import os
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from air_track.detector.engine import Predictor
from air_track.detector.data.dataset import DetectDataset
from air_track.detector.utils.detect_utils import combine_images, xywh2xyxy, check_boundary
from air_track.detector.visualization.visualize_and_save import draw_feature_img_orig_data, draw_feature_img
from air_track.utils import common_utils, reprod_init, col_data_save_csv


def process_input(cur_image, prev_images, device):
    """输入数据前处理"""
    # 将cur数据与prev数据合并
    input_data = combine_images(prev_images, cur_image)

    input_data = input_data.float().to(device)

    return input_data


def get_prev_step_frame(data, model_params):
    """
    从dataset中取出prev的数据

    存储顺序为: prev_frames = [index-frame_step、index-frame_step*2 ......]
    """
    prev_images = []

    for i in range(model_params['input_frames'] - 1):
        prev_images.append(data[f'prev_image_aligned{i}'])

    return prev_images


def check_pos_or_neg(label_feature):
    """判断是正/负样本"""
    max_value = label_feature.max()
    if max_value == 1:
        return 1
    else:
        return 0


def get_data_and_label(cfg, detected_objects, pred_feature, label_feature):
    """获取二级分类器的数据与标签"""
    second_classifier_params = cfg['second_classifier_params']
    down_scale = cfg['model_params']['down_scale']

    img_size_w, img_size_h = second_classifier_params['save_data_size']
    orig_h, orig_w = pred_feature.shape[-2:]

    img_list, label_list = [], []
    for detected_object in detected_objects:
        classify_object = copy.deepcopy(detected_object)
        classify_object['cx'] = detected_object['cx'] / down_scale
        classify_object['cy'] = detected_object['cy'] / down_scale
        # 以候选目标的中心为基准，裁剪出指定尺寸的特征图
        bbox_cxcywh = [classify_object['cx'].item(), classify_object['cy'].item(), img_size_w, img_size_h]
        bbox_xyxy = xywh2xyxy(bbox_cxcywh)

        # 检查边界条件并调整
        bbox_xyxy = check_boundary(bbox_xyxy, img_size_w, img_size_h, orig_w, orig_h)

        x1, y1, x2, y2 = bbox_xyxy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img = pred_feature[:, y1:y2, x1:x2]
        label = label_feature[y1:y2, x1:x2]

        # one-hot格式的标签
        label_one_hot = np.zeros(int(second_classifier_params['nb_classes']))
        # 判断是正/负样本
        idx = check_pos_or_neg(label)
        label_one_hot[idx] = 1

        img_list.append(img)
        label_list.append(label_one_hot)

    return img_list, label_list


def write_data_and_label(img_lists, label_lists, save_path, save_img_flag=True, max_pixel=255):
    """将数据与标签写入文件"""
    img_path = os.path.join(save_path, 'img')
    img_path_orig = os.path.join(save_path, 'img_orig')
    npy_path = os.path.join(save_path, 'npy')
    os.makedirs(npy_path, exist_ok=True)

    idx_list, neg_list, pos_list = [], [], []
    for i, (img, label) in enumerate(zip(img_lists, label_lists)):
        print(f"当前保存第{i + 1}/{len(img_lists)}帧")

        idx_list.append(i + 1)
        neg_list.append(label[0])
        pos_list.append(label[1])

        if save_img_flag:
            draw_feature_img_orig_data(img[-1], f'{img_path_orig}/{i + 1}.png', is_plt_star=False)
            draw_feature_img(img[-1], f'{img_path}/{i + 1}.png', max_pixel=max_pixel)
        np.save(os.path.join(npy_path, f'{i + 1}.npy'), img)

    save_labels = {'index': idx_list, 'neg': neg_list, 'pos': pos_list}
    col_data_save_csv(f'{save_path}/label.csv', save_labels)


def gen_classifier_data(predictor, stage):
    """主函数读取一个文件夹下的图像进行检测"""
    cfg = predictor.cfg

    model_params = cfg['model_params']
    second_classifier_params = cfg['second_classifier_params']
    save_path = second_classifier_params['save_path']
    save_frame_num = second_classifier_params['save_frame_num']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    dataset = DetectDataset(stage, cfg_data=cfg)
    dataset.gamma_flag = False
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        pin_memory=True,
        shuffle=False
    )

    img_lists, label_lists = [], []
    for i, data in enumerate(dataloader):
        print(f"当前处理第{i + 1}/{len(dataloader)}帧")
        cur_img = data['image']
        label_cls = data['cls']

        # 读取prev帧图像数据
        prev_images = get_prev_step_frame(data, model_params)

        input_data = process_input(cur_img, prev_images, device)

        # 进行检测，返回检测结果和推理耗时
        prediction, detected_objects, inference_time = predictor.predict(input_data)

        # 按照cfg['second_classifier_params']['save_data_size']裁剪特征图和设置标签
        mask_pred_feature = prediction[-1][0].float().cpu().detach().numpy()
        cls_label_feature = 1.0 - label_cls[0][0].float().cpu().detach().numpy()

        # 检出目标
        if detected_objects:
            img_list, label_list = get_data_and_label(cfg, detected_objects, mask_pred_feature, cls_label_feature)
            img_lists += img_list
            label_lists += label_list

        # 保存数据的最大数量
        if i + 1 >= float(save_frame_num):
            print('\n已达到设置的最大数据量，进行保存......\n')
            break

    write_data_and_label(img_lists, label_lists, save_path, save_img_flag=second_classifier_params['save_img_flag'],
                         max_pixel=second_classifier_params['max_pixel'])


if __name__ == "__main__":
    """有标签预测"""
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_yaml = os.path.join(script_dir, 'config/predict.yaml')
    classifier_yaml = os.path.join(script_dir, 'config/gen_classifier_data.yaml')
    cfg_predict = common_utils.combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    # 合并若干个yaml的配置文件内容
    yaml_list = [pred_yaml, classifier_yaml]

    # 创建预测器
    predictor = Predictor(yaml_list=yaml_list)
    predictor.set_model()

    # gen_classifier_data(predictor, stage=predictor.cfg['stage_train'])
    # gen_classifier_data(predictor, stage=predictor.cfg['stage_valid'])
    gen_classifier_data(predictor, stage=predictor.cfg['stage_test'])
