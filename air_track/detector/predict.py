import os
import cv2
from air_track.detector.engine import Predictor
from air_track.detector.visualization.visualize_and_save import visualize_and_save
from air_track.utils import common_utils, reprod_init, check_and_change_img_size


def predict_two_frame(predictor, cur_img_path, prev_img_path):
    """主函数读取一个文件夹下的图像进行检测"""
    cfg = predictor.cfg
    img_size_w, img_size_h = cfg['dataset_params']['img_size']

    cur_file = cur_img_path
    prev_file = prev_img_path

    cur_img = cv2.imread(cur_file, cv2.IMREAD_UNCHANGED)
    orig_h, orig_w = cur_img.shape[:2]
    cur_img = check_and_change_img_size(cur_img, img_size_w, img_size_h)
    prev_img = cv2.imread(prev_file, cv2.IMREAD_UNCHANGED)
    prev_img = check_and_change_img_size(prev_img, img_size_w, img_size_h)
    prev_images = [prev_img]

    # 数据前处理
    input_data = predictor.process_input(cur_img, prev_images)

    # 进行检测，返回检测结果和推理耗时
    prediction, detected_objects, inference_time = predictor.predict(input_data)

    # 检测结果可视化
    if cfg['visualize_flag']:
        # 可视化结果保存
        scale_w = img_size_w / orig_w
        scale_h = img_size_h / orig_h
        visualize_and_save(cur_file, detected_objects, cfg['visualize_save_dir'], scale_w, scale_h)

    print('Successful inference. ')


def predict_without_gt(predictor, img_path):
    """主函数读取一个文件夹下的图像进行检测"""
    cfg = predictor.cfg

    frame_step = cfg['frame_step']
    input_frames = cfg['input_frames']
    img_size_w, img_size_h = cfg['dataset_params']['img_size']

    # 超参数设置为小于2帧或者步长不大于0，则返回空列表
    if input_frames < 2 or frame_step <= 0:
        assert ValueError('input_frames < 2 or frame_step <= 0. ')

    file_names = os.listdir(img_path)
    file_names.sort()
    inference_times = []

    for i, file_name in enumerate(file_names):
        print(f"当前处理第{i}帧，名称为{file_name}")

        if i < frame_step * (input_frames - 1):
            continue
        cur_file = os.path.join(img_path, file_name)

        cur_img = cv2.imread(cur_file, cv2.IMREAD_UNCHANGED)
        orig_h, orig_w = cur_img.shape[:2]
        cur_img = check_and_change_img_size(cur_img, img_size_w, img_size_h)

        # 读取prev帧图像数据
        prev_images = predictor.get_prev_step_frame(img_path, file_names, i)

        # 数据前处理
        input_data = predictor.process_input(cur_img, prev_images)

        # 进行检测，返回检测结果和推理耗时
        prediction, detected_objects, inference_time = predictor.predict(input_data)
        inference_times.append(inference_time)

        # 检测结果可视化
        if cfg['visualize_flag']:
            # 可视化结果保存
            scale_w = img_size_w / orig_w
            scale_h = img_size_h / orig_h
            visualize_and_save(cur_file, detected_objects, cfg['visualize_save_dir'], scale_w, scale_h)

    inference_times.sort()
    inference_times = inference_times[2:-2]
    print('Speed inference: ', sum(inference_times) / len(inference_times), 'ms')


if __name__ == "__main__":
    """无标签预测"""
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_yaml = os.path.join(script_dir, 'config/predict.yaml')
    cfg_predict = common_utils.combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    model_dir = os.path.dirname(cfg_predict['model_path'])

    # 使用模型保存路径下的配置文件
    train_yaml = os.path.join(model_dir, 'hrnet_train.yaml')

    # yaml文件列表
    yaml_list = [train_yaml, pred_yaml]

    # 创建预测器
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = cfg_predict['device']
    predictor.set_model()

    '''两帧确定图像预测'''
    cur_img_path = '/home/csz_changsha/data/AR_2410/normal_airtrack/lz_trj_1_0300/Images/lz_trj_1_0300_result/User_channel_1_000508.png'
    prev_img_path = '/home/csz_changsha/data/AR_2410/normal_airtrack/lz_trj_1_0300/Images/lz_trj_1_0300_result/User_channel_1_000507.png'
    predict_two_frame(predictor, cur_img_path, prev_img_path)

    '''读取文件夹逐帧检测，但不使用标签'''
    files_path = '/home/csz_changsha/data/AR_2410/normal_airtrack/0730/Images/0730_result'
    predict_without_gt(predictor, img_path=files_path)
