import copy
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
from air_track.detector.engine.predictor import Predictor
from air_track.detector.data.dataset import DetectDataset
from air_track.detector.utils.analyse_yolo_data_distribution import analyse_bbox_distribution
from air_track.detector.utils.detect_utils import combine_images, xywh2xyxy, check_boundary, xyxy2xywh, cxcywhn2xyxy
from air_track.utils import common_utils, reprod_init, col_data_save_csv, round_up_to_nearest_power_of_two


def process_input(prev_images, cur_image, device):
    """输入数据前处理"""
    # 将cur数据与prev数据合并
    input_data = combine_images(prev_images, cur_image)

    input_data = input_data.float().to(device)

    return input_data


def get_prev_step_frame(data, dataset_params):
    """
    从dataset中取出prev的数据

    存储顺序为: prev_frames = [index-frame_step、index-frame_step*2 ......]
    """
    prev_images = []

    for i in range(dataset_params['input_frames'] - 1):
        prev_images.append(data[f'prev_image_aligned{i}'])

    return prev_images


def check_pos_or_neg(label_feature):
    """判断是正/负样本"""
    max_value = label_feature.max()
    if max_value == 1:
        return 1
    else:
        return 0


def get_enum_data_and_label(cfg, orig_img, label_feature, scale_w=1, scale_h=1):
    """在原图上遍历获取二级分类器的数据与标签"""
    second_classifier_params = cfg['second_classifier_params']
    down_scale = cfg['model_params']['down_scale']

    img_size_w, img_size_h = second_classifier_params['save_data_size']
    orig_h, orig_w = orig_img.shape[-2:]

    img_list, label_list = [], []
    for w in range(0, orig_w, img_size_w):
        for h in range(0, orig_h, img_size_h):
            x1, y1, x2, y2 = int(w / (down_scale * scale_w)), int(h / (down_scale * scale_h)), \
                             int((w + img_size_w) / (down_scale * scale_w)), int((h + img_size_h) / (down_scale * scale_h))
            img = orig_img[0, :, h:h + img_size_h, w:w + img_size_w].cpu().numpy()  # img为(c, h, w)
            label = label_feature[y1:y2, x1:x2]

            # one-hot格式的标签
            label_one_hot = np.zeros(int(second_classifier_params['nb_classes']))
            # 判断是正/负样本
            idx = check_pos_or_neg(label)
            label_one_hot[idx] = 1

            img_list.append(img)
            label_list.append(label_one_hot)

    return img_list, label_list


def get_data_and_label(cfg, detected_objects, orig_img, cur_labels, scale_w=1, scale_h=1):
    """在候选目标上获取二级分类器的数据与标签"""
    second_classifier_params = cfg['second_classifier_params']
    down_scale = cfg['model_params']['down_scale']

    per_frame_object_nums = second_classifier_params['per_frame_object_nums']
    orig_h, orig_w = orig_img.shape[1:3]

    pos_exist = False
    img_list, label_list = [], []
    img_size_w, img_size_h = None, None

    # 未检出目标
    if len(detected_objects) == 0:
        return img_list, label_list

    for detected_object in detected_objects:
        classify_object = copy.deepcopy(detected_object)
        classify_object['cx'] = detected_object['cx'] / down_scale
        classify_object['cy'] = detected_object['cy'] / down_scale

        if second_classifier_params['save_data_size'] != 'None':
            img_size_w, img_size_h = second_classifier_params['save_data_size']
        else:
            # 确定长边
            max_len = max(detected_object['w'], detected_object['h'])
            img_size_w = max_len * 2
            img_size_h = max_len * 2
            # 向上取整到最近的2的倍数
            img_size_w = round_up_to_nearest_power_of_two(img_size_w * scale_w)
            img_size_h = round_up_to_nearest_power_of_two(img_size_h * scale_h)

        # 以候选目标的中心为基准，裁剪出指定尺寸的图像(模型输入尺寸)
        bbox_xywh = [detected_object['cx'].item() * scale_w,
                     detected_object['cy'].item() * scale_h, img_size_w, img_size_h]
        bbox_xyxy = xywh2xyxy(bbox_xywh)

        # 检查边界条件并调整
        bbox_xyxy = check_boundary(bbox_xyxy, img_size_w, img_size_h, orig_w, orig_h)
        bbox_xywh = xyxy2xywh(bbox_xyxy)

        cx, cy = bbox_xywh[:2]
        x1, y1, x2, y2 = bbox_xyxy
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        pos_flag = False
        cls_name = 'other'
        for cur_label in cur_labels[0]:
            label_xywhn = [cur_label['cx'].item(), cur_label['cy'].item(), cur_label['w'].item(), cur_label['h'].item()]
            label_xyxy = cxcywhn2xyxy(label_xywhn, orig_w, orig_h)
            label_x1, label_y1, label_x2, label_y2 = label_xyxy
            if label_x1 <= cx <= label_x2 and label_y1 <= cy <= label_y2:
                pos_flag = True
                cls_name = cur_label['cls_name'][0]
                break
            else:
                continue

        if orig_img.shape[-1] % 3 == 0:
            img = orig_img[0, y1:y2, x1:x2, -3:].cpu().numpy()
        else:
            img = orig_img[0, y1:y2, x1:x2, -1:].cpu().numpy()  # img为(c, h, w)

        # one-hot格式的标签
        label_one_hot = np.zeros(int(second_classifier_params['nb_classes']) + 1, dtype=object)
        label_one_hot[0] = cls_name  # 第一个位置存字符串
        label_one_hot[1:] = 0  # 其他位置存数值
        # 判断是正/负样本
        if pos_flag:
            idx = -1
        else:
            idx = -2
        label_one_hot[idx] = 1

        # 判断检测器检测出来的目标是否存在真目标
        if label_one_hot[-1] == 1:
            pos_exist = True

        img_list.append(img)
        label_list.append(label_one_hot)

    '''如果检测器检测出来的目标不存在真目标，则从标签特征图中找到真目标的位置，并把真目标的img和label插入到列表开头'''
    if not pos_exist:
        for cur_label in cur_labels[0]:
            cls_name = cur_label['cls_name'][0]
            label_xywhn = [cur_label['cx'].item(), cur_label['cy'].item(), cur_label['w'].item(), cur_label['h'].item()]
            label_xyxy = cxcywhn2xyxy(label_xywhn, orig_w, orig_h)

            # 检查边界条件并调整
            label_xyxy = check_boundary(label_xyxy, img_size_w, img_size_h, orig_w, orig_h)
            x1, y1, x2, y2 = label_xyxy
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if orig_img.shape[-1] % 3 == 0:
                img = orig_img[0, y1:y2, x1:x2, -3:].cpu().numpy()
            else:
                img = orig_img[0:, y1:y2, x1:x2, -1].cpu().numpy()  # img为(c, h, w)

            label_one_hot = np.zeros(int(second_classifier_params['nb_classes']) + 1, dtype=object)
            label_one_hot[0] = cls_name  # 第一个位置存字符串
            label_one_hot[1:] = 0  # 其他位置存数值
            label_one_hot[-1] = 1

            # 在列表开头插入元素
            img_list.insert(0, img)
            label_list.insert(0, label_one_hot)
    else:  # 已经存在真目标，则将真目标移动到列表开头
        idx = None
        for i, arr in enumerate(label_list):
            if np.array_equal(arr[1:], np.array([0., 1.])):
                idx = i
                break
        if idx is not None:
            img_temp = img_list.pop(idx)
            label_temp = label_list.pop(idx)
            img_list.insert(0, img_temp)
            label_list.insert(0, label_temp)

    '''如果img和label列表数量大于object_nums，则将置信度最低的一个样本（即最后一个）移动到列表开头，最终只保留object_nums个样本'''
    if len(img_list) > float(per_frame_object_nums):
        # 将最后一个元素移动到第一个位置
        last_element = img_list.pop()  # 移除并返回最后一个元素
        img_list.insert(0, last_element)  # 在列表开头插入最后一个元素
        last_element = label_list.pop()  # 移除并返回最后一个元素
        label_list.insert(0, last_element)  # 在列表开头插入最后一个元素

        img_list = img_list[:per_frame_object_nums]
        label_list = label_list[:per_frame_object_nums]

    return img_list, label_list


def write_img(data_idx, img_lists, label_lists, save_path, save_img_flag=True, max_pixel=255, use_gray=False):
    """将数据写入文件"""
    img_path = os.path.join(save_path, 'img')
    npy_path = os.path.join(save_path, 'npy')
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(npy_path, exist_ok=True)

    for i in range(data_idx, len(label_lists)):
        print(f"当前图像保存第{i + 1}/{len(label_lists)}帧")
        img = img_lists[i - data_idx]

        if save_img_flag:
            cv2.imwrite(f'{img_path}/{i + 1}.png', img)
        np.save(os.path.join(npy_path, f'{i + 1}.npy'), img)


def write_label(label_lists, save_path):
    """将标签写入文件"""
    os.makedirs(save_path, exist_ok=True)

    idx_list, cls_name_list, neg_list, pos_list = [], [], [], []
    for i, label in enumerate(label_lists):
        print(f"当前标签保存第{i + 1}/{len(label_lists)}帧")
        label = label_lists[i]

        idx_list.append(i + 1)
        cls_name_list.append(label[0])
        neg_list.append(label[1])
        pos_list.append(label[2])

    save_labels = {'index': idx_list, 'cls_name': cls_name_list, 'neg': neg_list, 'pos': pos_list}
    col_data_save_csv(f'{save_path}/label.csv', save_labels)


def gen_classifier_data(predictor, stage):
    """主函数读取一个文件夹下的图像进行检测"""
    cfg = predictor.cfg

    dataset_params = cfg['dataset_params']
    img_read_method = predictor.dataset_params['img_read_method'].lower()
    if img_read_method == 'gray':
        use_gray = True
    else:
        use_gray = False
    second_classifier_params = cfg['second_classifier_params']
    save_path = os.path.join(second_classifier_params['save_path'], stage)
    save_frame_num = second_classifier_params['save_frame_num']
    device = predictor.device

    # 生成二级分类器数据专属配置
    cfg['dataset_params']['frame_align_flag'] = False
    cfg['dataset_params']['gamma_flag'] = False

    if cfg['dataset_params']['input_frames'] > 1:
        shuffle = False
    else:
        shuffle = True

    dataset = DetectDataset(stage, cfg_data=cfg)
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        pin_memory=True,
        shuffle=shuffle
    )

    data_idx = 0
    img_lists, label_lists = [], []
    for i, data in enumerate(dataloader):
        print(f"当前处理第{i + 1}/{len(dataloader)}帧")
        cur_img = data['cur_image']
        cur_image_orig = data['cur_img_orig']  # 原始图像数据
        cur_labels = data['cur_labels']

        scale_w = cur_image_orig.shape[2] / cur_img.shape[-1]
        scale_h = cur_image_orig.shape[1] / cur_img.shape[-2]

        # 读取prev帧图像数据
        prev_images = get_prev_step_frame(data, dataset_params)

        input_data = predictor.process_multi_frame_input(prev_images, cur_img)

        # 进行检测，返回检测结果和推理耗时
        prediction, detected_objects, inference_time = predictor.predict(input_data)

        # 在原图上遍历抠图
        # img_list, label_list = get_enum_data_and_label(cfg, input_data, mask_label_feature)
        # img_lists += img_list
        # label_lists += label_list

        # 检出目标
        if detected_objects:
            img_list, label_list = get_data_and_label(cfg, detected_objects, cur_image_orig, cur_labels, scale_w, scale_h)
            img_lists += img_list
            label_lists += label_list

        # 保存数据的最大数量
        if i + 1 >= float(save_frame_num):
            print('\n已达到设置的最大数据量，进行保存......\n')
            break

        if i % 100 == 0:
            write_img(data_idx, img_lists, label_lists, save_path,
                      save_img_flag=second_classifier_params['save_img_flag'],
                      max_pixel=second_classifier_params['max_pixel'], use_gray=use_gray)
            data_idx = len(label_lists)
            img_lists = []

    # 把最后剩余的写入进去
    write_img(data_idx, img_lists, label_lists, save_path,
              save_img_flag=second_classifier_params['save_img_flag'],
              max_pixel=second_classifier_params['max_pixel'], use_gray=use_gray)
    write_label(label_lists, save_path)


if __name__ == "__main__":
    """有标签预测"""
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    predict_yaml = os.path.join(script_dir, 'config/predict_dalachi.yaml')
    classifier_yaml = os.path.join(script_dir, 'config/gen_classifier_data.yaml')
    cfg_predict = common_utils.combine_load_cfg_yaml(yaml_paths_list=[predict_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    # 合并若干个yaml的配置文件内容
    yaml_list = [predict_yaml, classifier_yaml]

    # 创建预测器
    predictor = Predictor(yaml_list=yaml_list)

    # 分析bbox分布
    img_shape = cfg_predict['dataset_params']['img_size']
    try:
        # TODO 目前分析bbox分支只支持yolo格式数据，先采用try模式
        bbox_distribution = analyse_bbox_distribution(cfg_predict, img_shape)
        max_w = bbox_distribution['width_distribution']['98%']
        max_h = bbox_distribution['height_distribution']['98%']
    except:
        print("分析bbox分布失败，使用'gen_classifier_data.yaml'配置文件中默认值")
        # max_w, max_h = predictor.cfg['second_classifier_params']['save_data_size']

    # 向上取整到最近的2的倍数
    # max_w = round_up_to_nearest_power_of_two(max_w)
    # max_h = round_up_to_nearest_power_of_two(max_h)

    # 确定最大边长
    # max_side = max(max_w, max_h)

    # 可选自动计算二级分类器数据尺寸
    # predictor.cfg['second_classifier_params']['save_data_size'] = [max_side, max_side]
    print('二级分类器数据尺寸 w、h: ', predictor.cfg['second_classifier_params']['save_data_size'])

    predictor.set_model()

    stage_list = [cfg_predict['stage_train']]
    # stage_list = [cfg_predict['stage_train'], cfg_predict['stage_valid'], cfg_predict['stage_test']]
    for stage in stage_list:
        gen_classifier_data(predictor, stage)
