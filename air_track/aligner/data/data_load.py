import os
import pandas as pd
from air_track.utils import img_fn, common_utils


# pd的行列配置
pd.set_option('display.max_columns', None)  # 不限制列数
pd.set_option('display.max_rows', None)  # 不限制行数


def write_single_part_data(cfg, part, label_file, cache_fn):
    """将标签数据写入缓存文件"""
    os.makedirs(os.path.dirname(cache_fn), exist_ok=True)

    frames_dict = {}
    df = pd.read_csv(label_file)

    # 以行为索引，遍历每行
    for _, row in sorted(df.iterrows()):
        flight_id = row['flight_id']
        img_name = row['img_name'][:-4]  # 截至到-4是为了舍去.png
        frame_num_id = row['frame']

        img_file = img_fn(cfg, part, flight_id, img_name)
        # 图片不存在
        if not os.path.exists(img_file):
            print('skip missing img: ', img_file)
            continue

        # 缓存文件存储内容
        key = (flight_id, frame_num_id)
        if key not in frames_dict:
            frames_dict[key] = dict(
                part=part,
                flight_id=flight_id,
                frame_num_id=frame_num_id,
                img_name=img_name,
                img_path=img_file
            )

    # 写入pkl文件
    if len(frames_dict) > 0:
        frames = pd.DataFrame([frames_dict[k] for k in sorted(frames_dict.keys())])
        frames.to_pickle(cache_fn)


def read_single_part_data(cfg, part, cache_fn) -> pd.DataFrame:
    """读取当前part的数据集缓存文件"""
    data_dir = cfg['data_dir']

    # 文件不存在需先写入
    if not os.path.exists(cache_fn):
        label_name = cfg['label_file']
        label_file = f'{data_dir}/{part}/{label_name}'
        write_single_part_data(cfg, part, label_file, cache_fn)

    # 从缓存文件读
    frames = pd.read_pickle(cache_fn)
    print("frames: ", type(frames), len(frames))

    return frames


def load_datasets(cfg, parts, stage):
    """根据 parts 读取整个训练集或者验证集"""
    frames = []
    for part in parts:
        with common_utils.timeit_context('load dataset_part: ' + str(part)):
            # 缓存文件
            cache_data_dir = cfg['cache_data_dir']
            cache_fn = f'{cache_data_dir}/ds_align_{part}.pkl'
            print(cache_fn)
            frames.append(read_single_part_data(cfg, part, cache_fn))

    # 将几个part的frames拼接起来
    frames = pd.concat(frames, axis=0, ignore_index=True).reset_index(drop=True)

    print('当前模式为：', stage)
    print(f'当前 {stage} 下的数据集数量为：', len(frames))

    return frames


if __name__ == "__main__":
    """用于测试阶段使用，现作为展示代码，若需单独使用，将下段代码拿出去到单独脚本中使用，莫要修改此文件"""
    cfg_path = '../config/dataset.yaml'
    cfg_data = common_utils.load_yaml(cfg_path)

    frames = load_datasets(cfg=cfg_data, parts=cfg_data['part_train'], stage=cfg_data['stage_train'])
