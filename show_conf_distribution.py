import os
import cv2
import pickle
from air_track.detector.utils.calculate_metrics import calculate_metrics
from air_track.detector.visualization.visualize_and_save import show_bar_chart
from air_track.utils import combine_load_cfg_yaml, reprod_init

if __name__ == '__main__':
    """可视化整个测试数据的置信度分布"""

    '''第一种方法：直接读取全流程测试的缓存文件'''
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inference_yaml = os.path.join(script_dir, 'air_track/config/inference.yaml')
    cfg_inference = combine_load_cfg_yaml(yaml_paths_list=[inference_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_inference['seed'])

    # 进行测试
    dataset_params = cfg_inference['dataset_params']

    cache_file = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack_all/AirTrack_single_img_predict/output_malan/纯一级检测器_max_obj_10_conf_0.01/cache_test.pkl'
    save_path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack_all/AirTrack_single_img_predict/output_malan/纯一级检测器_max_obj_10_conf_0.01'

    # 使用 'rb' 模式打开文件进行读取（'rb' 代表 'read binary'）
    with open(cache_file, 'rb') as file:
        # 使用 pickle.load 来读取文件中的数据
        data = pickle.load(file)

    # cur_files = data['cur_files']
    bbox_gts = data['bbox_xyxy']
    candidate_targets = data['candidate_targets']
    frame_nums = data['frame_nums']
    scale_w = data['scale_w']
    scale_h = data['scale_h']
    detection_rate, pos_pred_x, pos_conf_list, neg_conf_list, pos_candidate_targets, neg_candidate_targets = \
        calculate_metrics(dataset_params['classes'], bbox_gts, candidate_targets, frame_nums, scale_w, scale_h,
                          threshold=0.25, target_min_size=10, iou=True)

    '''第二种方法：直接读取标签处的置信度缓存文件'''
    # cache_file = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack_all/AirTrack_single_img_predict/output_malan/纯一级检测器_max_obj_10_conf_0.01/cache_test.pkl'
    # save_path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack_all/AirTrack_single_img_predict/output'
    #
    # # 使用 'rb' 模式打开文件进行读取（'rb' 代表 'read binary'）
    # with open(cache_file, 'rb') as file:
    #     # 使用 pickle.load 来读取文件中的数据
    #     data = pickle.load(file)
    #
    # frame_nums = data['frame_nums']
    # scale_w = data['scale_w']
    # scale_h = data['scale_h']
    # pos_conf_list = data['pos_conf_list']

    show_bar_chart(pos_conf_list, save_path, file_name='Confidence_distribution.png',
                   range_min=0, range_max=0.101, range_interval=0.005,
                   # range_min=0, range_max=1, range_interval=0.05,
                   title='Confidence distribution', xlabel='Confidence', ylabel='Count/Scale')
