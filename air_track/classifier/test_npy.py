import os
import numpy as np
import pandas as pd
from air_track.classifier.engine import Predictor
from air_track.utils import combine_load_cfg_yaml, extract_number, reprod_init


def predict_with_gt(predictor, data_dir):
    cfg = predictor.cfg

    npy_folder = os.path.join(data_dir, cfg['npy_folder'])
    label_file = os.path.join(data_dir, cfg['label_file'])

    npy_paths = [f for f in os.listdir(npy_folder) if f.endswith(cfg['npy_format'])]
    npy_paths.sort(key=extract_number)
    labels = pd.read_csv(label_file).values

    if len(npy_paths) != len(labels):
        assert 'frame_num and label_num not equal !'

    num_correct = 0
    pos_cnt = 0
    pos_classify_neg_cnt = 0
    neg_classify_pos_cnt = 0
    inference_times = []
    frame_num = len(npy_paths)
    for i, (npy_file_name, label) in enumerate(zip(npy_paths, labels)):
        print(f'当前处理第{i + 1}/{frame_num}帧')
        npy_file = os.path.join(npy_folder, npy_file_name)
        data = np.load(npy_file)
        label = cfg['classes'][list(label[1:]).index(1)]

        data = predictor.process_input(data)
        classify_prediction, cls_name_batch, inference_time = predictor.predict(data)
        inference_times.append(inference_time)

        if label == 'pos':
            pos_cnt += 1

        if cls_name_batch[0] == label:
            num_correct += 1
        elif label == 'pos' and cls_name_batch[0] == 'neg':
            pos_classify_neg_cnt += 1
        elif label == 'neg' and cls_name_batch[0] == 'pos':
            neg_classify_pos_cnt += 1

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

    predict_with_gt(predictor, cfg_predict['test_data_dir'])
