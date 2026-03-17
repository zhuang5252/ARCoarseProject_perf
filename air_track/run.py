import os
from air_track.detector.model.model import Model
from air_track.detector.engine.predictor import Predictor
from air_track.aligner.special_test.predict_multi_offset import predict_folder_offsets, predict_aot_folder_offsets
from air_track.utils import combine_load_cfg_yaml, reprod_init, load_model


def train_aligner(dataset_yaml, train_yaml):
    """训练帧对齐模型"""
    from air_track.aligner.train import train

    print('帧对齐模型训练中······')

    # 读取yaml文件
    yaml_list = [dataset_yaml, train_yaml]
    # 合并若干个yaml的配置文件内容
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    model_path = train(cfg=cfg_data, yaml_list=yaml_list)

    print(f'帧对齐模型训练完毕，模型路径为: {model_path}')

    return model_path


def gen_yolo_aligner_offset(model_path, predict_yaml):
    """使用帧对齐模型生成offset"""
    from air_track.aligner.engine.predictor_return_dx_dy_angle import Predictor

    print('帧对齐模型生成offset中······')

    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[predict_yaml])

    yaml_list = [predict_yaml]
    predictor = Predictor(yaml_list=yaml_list)
    predictor.model_path = model_path
    predictor.device = cfg_predict['device']
    predictor.set_model()

    data_dir = cfg_predict['data_dir']
    parts = cfg_predict['part_train'] + cfg_predict['part_val'] + cfg_predict['part_test']
    for part in parts:
        test_data_dir = os.path.join(data_dir, part)
        if os.path.isdir(test_data_dir):
            predict_folder_offsets(predictor, part)

    print('帧对齐模型生成offset完毕')



def gen_aot_aligner_offset(model_path, predict_yaml):
    """使用帧对齐模型生成offset"""
    from air_track.aligner.engine.predictor_return_dx_dy_angle import Predictor

    print('帧对齐模型生成offset中······')

    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[predict_yaml])

    yaml_list = [predict_yaml]
    predictor = Predictor(yaml_list=yaml_list)
    predictor.model_path = model_path
    predictor.device = cfg_predict['device']
    predictor.set_model()

    data_dir = cfg_predict['data_dir']
    parts = cfg_predict['part_train'] + cfg_predict['part_val'] + cfg_predict['part_test']
    for part in parts:
        test_data_dir = os.path.join(data_dir, part)
        if os.path.isdir(test_data_dir):
            predict_aot_folder_offsets(predictor, part)

    print('帧对齐模型生成offset完毕')



def train_detector(dataset_yaml, train_yaml):
    """训练一级检测器"""
    from air_track.detector.train import train

    print('一级检测器模型训练中······')

    # 读取yaml文件
    yaml_list = [dataset_yaml, train_yaml]
    # 合并若干个yaml的配置文件内容
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    model_path = train(cfg=cfg_data, yaml_list=yaml_list)

    print(f'一级检测器模型训练完毕，模型路径为: {model_path}')

    return model_path


def gen_classifier_dataset(model_path, predict_yaml, classifier_yaml):
    """使用一级检测器生成分类器数据"""
    from air_track.detector.special_test.gen_classifier_data import gen_classifier_data

    print('使用一级检测器生成分类器数据中······')

    cfg_predict = combine_load_cfg_yaml(yaml_paths_list=[predict_yaml])

    # 合并若干个yaml的配置文件内容
    yaml_list = [predict_yaml, classifier_yaml]

    # 创建预测器
    predictor = Predictor(yaml_list=yaml_list)

    model_params = predictor.model_params
    model = Model(cfg=model_params, pretrained=model_params['pretrained'])
    predictor.model = load_model(model, model_path, device=predictor.device)

    stage_list = [cfg_predict['stage_train'], cfg_predict['stage_valid'], cfg_predict['stage_test']]

    for stage in stage_list:
        gen_classifier_data(predictor, stage)

    print('使用一级检测器生成分类器数据完毕')

    return predictor.cfg['second_classifier_params']['save_data_size']


def train_classifier(dataset_yaml, train_yaml, img_size):
    """训练二级分类器"""
    from air_track.classifier.train import train

    print('二级分类器模型训练中······')

    # 读取yaml文件
    yaml_list = [dataset_yaml, train_yaml]
    # 合并若干个yaml的配置文件内容
    cfg_data = combine_load_cfg_yaml(yaml_paths_list=yaml_list)

    model_path = train(cfg=cfg_data, yaml_list=yaml_list)

    print(f'二级分类器模型训练完毕，模型路径为: {model_path}')

    return model_path


if __name__ == '__main__':
    # 固定随机数种子
    reprod_init(seed=128)
    # 获取当前脚本所在的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    """帧对齐训练"""
    # aligner_dataset_yaml = os.path.join(script_dir, 'aligner/config/dataset_dabaige.yaml')
    # aligner_train_yaml = os.path.join(script_dir, 'aligner/config/align_train.yaml')
    # aligner_model_path = train_aligner(aligner_dataset_yaml, aligner_train_yaml)
    #
    # """帧对齐生成offset"""
    # aligner_predict_yaml = os.path.join(script_dir, 'aligner/config/predict_multi_offset.yaml')
    # gen_yolo_aligner_offset(aligner_model_path, aligner_predict_yaml)

    """一级检测器训练"""
    detector_dataset_yaml = os.path.join(script_dir, 'detector/config/dataset_dalachi.yaml')
    detector_train_yaml = os.path.join(script_dir, 'detector/config/detect_train_dalachi.yaml')
    detector_model_path = train_detector(detector_dataset_yaml, detector_train_yaml)

    """一级检测器生成二级分类器数据集"""
    detector_predict_yaml = os.path.join(script_dir, 'detector/config/predict_dalachi.yaml')
    gen_classifier_yaml = os.path.join(script_dir, 'detector/config/gen_classifier_data.yaml')
    classifier_data_size = gen_classifier_dataset(detector_model_path, detector_predict_yaml, gen_classifier_yaml)

    """二级分类器训练"""
    classifier_dataset_yaml = os.path.join(script_dir, 'classifier/config/dataset.yaml')
    classifier_train_yaml = os.path.join(script_dir, 'classifier/config/classify_train.yaml')
    classifier_model_path = train_classifier(classifier_dataset_yaml, classifier_train_yaml, classifier_data_size)

    print('二级分类器数据尺寸 w、h: ', classifier_data_size)
    print('一级检测器模型路径为: ', detector_model_path)
    print('二级分类器模型路径为: ', classifier_model_path)
