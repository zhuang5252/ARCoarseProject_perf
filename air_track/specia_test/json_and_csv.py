# -*- coding: utf-8 -*-
# @Author    : 
# @File      : AirTrack - json_2_csv.py
# @Created   : 2025/08/19 15:55
# @Desc      : 数据配置文件转换函数，json_to_csv、csv_to_json

import csv
import json
from pathlib import Path
from collections import defaultdict
from air_track.utils import load_csv


def json_to_csv(json_path: str, csv_path: str):
    """
    将JSON配置文件转换为CSV格式

    参数:
        json_path: 输入的JSON文件路径
        csv_path: 输出的CSV文件路径
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 准备CSV数据
    csv_rows = []

    # 遍历所有part类型(train/val/test)
    for part_name, part_data in data.items():
        if not part_data:  # 跳过空的部分(如part_test)
            continue

        # 遍历每个模型配置
        for model_name, config in part_data.items():
            # 处理可能的引号问题(如"'JDModel_0727_JD_201_second")
            clean_model_name = model_name.strip("'")

            csv_rows.append({
                "part": part_name,
                "data_name": clean_model_name,
                "nums": config["nums"],
                "shuffle": config["shuffle"]
            })

    # 写入CSV文件
    if csv_rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["part", "data_name", "nums", "shuffle"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV文件已生成: {Path(csv_path).resolve()}")
    else:
        print("警告: 未找到有效数据")


def csv_to_json(csv_path: str, json_path: str):
    """
    将CSV配置文件转换为JSON格式

    参数:
        json_path: 输入的JSON文件路径
        csv_path: 输出的CSV文件路径
    """
    # 初始化数据结构
    result = {
        "part_train": defaultdict(dict),
        "part_val": defaultdict(dict),
        "part_test": defaultdict(dict)
    }

    with open(csv_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            part = row['part']
            data_name = row['data_name']
            nums = int(row['nums'])
            shuffle = row['shuffle'].lower() == 'true'

            # 将数据添加到对应的部分
            result[part][data_name] = {
                "nums": nums,
                "shuffle": shuffle
            }

    # 将defaultdict转换为普通dict
    for part in result:
        result[part] = dict(result[part])

    # 写入JSON文件
    with open(json_path, mode='w', encoding='utf-8') as json_file:
        json.dump(result, json_file, indent=2, ensure_ascii=False)

    print(f"json文件已生成: {Path(json_path).resolve()}")


# 使用示例
if __name__ == "__main__":
    input_json = "/media/linana/1276341C76340351/csz/github/AirTrack/air_track/detector/config/template_dataset.json"  # 替换为你的JSON文件路径
    output_csv = "/media/linana/1276341C76340351/csz/github/AirTrack/air_track/detector/config/template_dataset.csv"
    json_to_csv(input_json, output_csv)
    # csv_to_json(output_csv, input_json)

    data = load_csv(output_csv)
    print()
