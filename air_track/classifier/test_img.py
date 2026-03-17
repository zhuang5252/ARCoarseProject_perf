import os
import cv2
import pandas as pd
from air_track.classifier.engine import Predictor
from air_track.detector.visualization.visualize_and_save import show_bar_chart
from air_track.utils import combine_load_cfg_yaml, extract_number, reprod_init


# TODO 需新增无标签预测
def predict_with_gt(predictor, data_dir):
    cfg = predictor.cfg

    img_folder = os.path.join(data_dir, cfg['img_folder'])
    label_file = os.path.join(data_dir, cfg['label_file'])
    img_read_method = predictor.img_read_method

    img_paths = [f for f in os.listdir(img_folder) if f.endswith(cfg['img_format'])]
    img_paths.sort(key=extract_number)
    labels = pd.read_csv(label_file).values

    if len(img_paths) != len(labels):
        assert 'frame_num and label_num not equal !'

    num_correct = 0
    pos_cnt = 0
    pos_classify_neg_cnt = 0
    neg_classify_pos_cnt = 0
    inference_times = []
    pos_conf_list, neg_conf_list = [], []
    frame_num = len(img_paths)
    for i, (img_file_name, label) in enumerate(zip(img_paths, labels)):
        print(f'当前处理第{i + 1}/{frame_num}帧')
        img_file = os.path.join(img_folder, img_file_name)

        if not os.path.exists(img_file):
            continue

        if img_read_method == 'gray':
            data = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        elif img_read_method == 'unchanged':
            data = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        else:
            data = cv2.imread(img_file)

        label = cfg['classes'][list(label[2:]).index(1)]

        data = predictor.process_input(data)
        classify_prediction, cls_name_batch, inference_time = predictor.predict(data)
        inference_times.append(inference_time)

        if label == 'pos':
            pos_cnt += 1
            pos_conf_list.append(classify_prediction[0][1])
        else:
            neg_conf_list.append(classify_prediction[0][0])

        if cls_name_batch[0] == label:
            num_correct += 1
        elif label == 'pos' and cls_name_batch[0] == 'neg':
            pos_classify_neg_cnt += 1
        elif label == 'neg' and cls_name_batch[0] == 'pos':
            neg_classify_pos_cnt += 1

    # 可视化正负样本概率分布
    show_bar_chart(pos_conf_list, data_dir, file_name='Pos_Confidence_distribution.png',
                   range_min=0, range_max=1.001, range_interval=0.05,
                   title='Confidence distribution', xlabel='Confidence', ylabel='Count/Scale')
    show_bar_chart(neg_conf_list, data_dir, file_name='Neg_Confidence_distribution.png',
                   range_min=0, range_max=1.001, range_interval=0.05,
                   title='Confidence distribution', xlabel='Confidence', ylabel='Count/Scale')

    neg_cnt = frame_num - pos_cnt
    acc = num_correct / frame_num
    pos_classify_pos_cnt = pos_cnt - pos_classify_neg_cnt
    neg_classify_neg_cnt = neg_cnt - neg_classify_pos_cnt

    print(f'accuracy: {acc}')
    print(f'pos_cnt: {pos_cnt}')
    print(f'neg_cnt: {neg_cnt}')
    print(f'pos_classify_pos_cnt: {pos_classify_pos_cnt}')
    print(f'pos_classify_neg_cnt: {pos_classify_neg_cnt}')

    print(f'neg_classify_neg_cnt: {neg_classify_neg_cnt}')
    print(f'neg_classify_pos_cnt: {neg_classify_pos_cnt}')

    print(f'pos_classify_pos_cnt / pos_cnt: {pos_classify_pos_cnt / pos_cnt}')
    print(f'pos_classify_neg_cnt / pos_cnt: {pos_classify_neg_cnt / pos_cnt}')

    print(f'neg_classify_neg_cnt / neg_cnt: {neg_classify_neg_cnt / neg_cnt}')
    print(f'neg_classify_pos_cnt / neg_cnt: {neg_classify_pos_cnt / neg_cnt}')
    inference_times = inference_times[2:-2]
    print('Speed inference: ', sum(inference_times) / len(inference_times), 'ms')


if __name__ == '__main__':
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pred_yaml = os.path.join(script_dir, 'config/predict.yaml')
    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[pred_yaml])

    # 固定随机数种子
    reprod_init(seed=cfg_predict['seed'])

    yaml_list = [pred_yaml]
    predictor = Predictor(yaml_list=yaml_list)
    predictor.device = cfg_predict['device']
    predictor.set_model()

    '''使用test_data_dir来进行推理测试'''
    predict_with_gt(predictor, cfg_predict['test_data_dir'])

    '''加载所有part来进行推理测试'''
    # parts = cfg_predict['parts']
    # for part in parts:
    #     test_data_dir = os.path.join(cfg_predict['data_dir'], part)
    #     predict_with_gt(predictor, test_data_dir)
