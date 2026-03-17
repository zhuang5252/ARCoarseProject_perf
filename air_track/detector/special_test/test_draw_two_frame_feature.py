import os
import pickle
import cv2
import torch
from air_track.detector.model import hrnet
from air_track.detector.utils.detect_utils import combine_images, argmax2d
from air_track.detector.visualization.visualize_and_save import draw_feature_img_orig_data
from air_track.utils import common_utils, load_model, reprod_init, check_and_change_img_size, numpy_2_tensor


def build_model(cfg):
    model_params = cfg['model_params']
    model = hrnet.__dict__[model_params['model_cls']](cfg=model_params, pretrained=False)
    return model


def process_input(cur_image, prev_images, max_pixel, device):
    """输入数据前处理"""
    cur_tensor_img = numpy_2_tensor(cur_image, max_pixel).unsqueeze(0)

    # prev_images可能不止一帧
    prev_tensor_images = []
    for prev_img in prev_images:
        prev_tensor_img = numpy_2_tensor(prev_img, max_pixel).unsqueeze(0)
        prev_tensor_images.append(prev_tensor_img)

    # 将cur数据与prev数据合并
    input_data = combine_images(prev_tensor_images, cur_tensor_img)

    input_data = input_data.float().to(device)

    return input_data


def process_output(pred, target_min_size=2, save_path='output'):
    # pred = [size, offset, distance, tracking, mask]
    # pred = {
    #     'cls': pred[0],
    #     'size': pred[1],
    #     'offset': pred[2],
    #     'distance': pred[3],
    #     'tracking': pred[4],
    #     'mask': pred[5],  # 按照当前模型设置，始终存在mask的输出
    # }
    pred = {
        'size': pred[0],
        'offset': pred[1],
        'distance': pred[2],
        'tracking': pred[3],
        'mask': pred[4],  # 按照当前模型设置，始终存在mask的输出
    }

    # pred结果包含5部分输出：size、offset、distance、tracking、mask
    pred['mask'] = torch.sigmoid(pred['mask'][0].float())  # torch.Size([1, 256, 320])

    for k in list(pred.keys()):
        pred[k] = pred[k][0].float().cpu().detach().numpy()

    pkl_path = os.path.join(save_path, 'model_pred_data')
    os.makedirs(pkl_path, exist_ok=True)
    max_iloc_y, max_iloc_x = argmax2d(pred['mask'])
    for k in list(pred.keys()):
        with open(pkl_path + f'/model_{k}_{target_min_size}_{target_min_size}.pkl', 'wb') as file:
            pickle.dump(pred[k], file)
        draw_feature_img_orig_data(pred[k],
                                   save_path + f'/model_pred_with_star/model_{k}_{target_min_size}_{target_min_size}.png',
                                   is_plt_star=True, max_iloc_x=max_iloc_x, max_iloc_y=max_iloc_y)
        draw_feature_img_orig_data(pred[k],
                                   save_path + f'/model_pred_without_star/model_{k}_{target_min_size}_{target_min_size}.png',
                                   is_plt_star=False, max_iloc_x=max_iloc_x, max_iloc_y=max_iloc_y)


def detect(model, cfg, cur_image, prev_images, device='cpu', save_path='output'):
    """模型检测"""
    model.eval()

    # False禁用梯度跟踪
    torch.set_grad_enabled(False)

    # 输入数据前处理
    input_data = process_input(cur_image, prev_images, cfg['max_pixel'], device)

    # 模型预测
    if cfg['half']:
        model = model.half()
        input_data = input_data.half()
        pred = model(input_data)
    else:
        # 自动混合精度计算
        # with torch.cuda.amp.autocast():
        pred = model(input_data)

    # 输出数据后处理
    process_output(pred, target_min_size=cfg['target_min_size'], save_path=save_path)


def get_prev_step_frame(path, file_names, index, frame_step, input_frames, img_size_w, img_size_h):
    """
    从file_names中取出prev的数据

    存储顺序为: prev_frames = [index-frame_step、index-frame_step*2 ......]
    """
    prev_images = []

    for i in range(frame_step, input_frames, frame_step):
        interval_index = i * frame_step

        file_name = file_names[index - interval_index]
        file = os.path.join(path, file_name)

        prev_img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        prev_img = check_and_change_img_size(prev_img, img_size_w, img_size_h)

        prev_images.append(prev_img)

    return prev_images


def predict_with_draw_feature(cfg, prev_file, cur_file, draw_feature_save_path):
    """主函数读取一个文件夹下的图像进行检测"""
    reprod_init(seed=cfg['seed'])

    model_path = cfg['model_path']
    img_size_w, img_size_h = cfg['dataset_params']['img_size']
    device = cfg.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

    # 定义及load模型
    model = build_model(cfg)
    model = load_model(model, model_path, device)

    cur_img = cv2.imread(cur_file, cv2.IMREAD_UNCHANGED)
    cur_img = check_and_change_img_size(cur_img, img_size_w, img_size_h)
    prev_image = cv2.imread(prev_file, cv2.IMREAD_UNCHANGED)
    prev_image = check_and_change_img_size(prev_image, img_size_w, img_size_h)
    prev_images = [prev_image]

    # 进行检测
    detect(model, cfg, cur_img, prev_images, device=device, save_path=draw_feature_save_path)


if __name__ == "__main__":
    """有标签预测"""
    train_yaml = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/model_saved/HRNet48/2024_11_29_18/hrnet48_train.yaml'
    pred_yaml = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/model_saved/HRNet48/2024_11_29_18/predict.yaml'

    # 合并若干个yaml的配置文件内容
    yaml_list = [train_yaml, pred_yaml]
    cfg_data = common_utils.combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    # 固定随机数种子
    reprod_init(cfg_data['seed'])

    prev_file = '/home/csz_changsha/data/AR_2410/normal_airtrack_test/sah_trj_2_1930/Images/sah_trj_2_1930_result/User_channel_1_000487.png'
    cur_file = '/home/csz_changsha/data/AR_2410/normal_airtrack_test/sah_trj_2_1930/Images/sah_trj_2_1930_result/User_channel_1_000488.png'
    save_path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/visual_feature_changed_labels_test'

    predict_with_draw_feature(cfg_data, prev_file, cur_file, save_path)
    print('Draw feature successful')
