import json
import os
import csv


def classify_samples_from_txt(labels_folder, output_csv, file_end='.txt'):
    # 准备结果列表
    results = []

    # 遍历labels文件夹中的所有txt文件
    for filename in os.listdir(labels_folder):
        if filename.endswith(file_end):
            txt_path = os.path.join(labels_folder, filename)

            # 检查txt文件是否为空
            is_empty = os.path.getsize(txt_path) == 0
            sample_type = 'negative' if is_empty else 'positive'

            # 添加到结果列表
            results.append({
                'filename': filename,
                'sample_type': sample_type
            })

    # 写入CSV文件
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'sample_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)


def classify_samples_from_json(labels_folder, output_csv):
    results = []

    for filename in os.listdir(labels_folder):
        if filename.endswith('.json'):
            json_path = os.path.join(labels_folder, filename)

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                # 检查 JSON 是否包含有效标注（假设标注存储在 'annotations' 或 'labels' 字段）
                is_empty = (
                        not data.get('annotations', [])  # 如果 'annotations' 不存在或为空
                        or not data.get('labels', [])  # 或者 'labels' 不存在或为空
                )

                sample_type = 'negative' if is_empty else 'positive'

                results.append({
                    'filename': filename,
                    'sample_type': sample_type
                })

            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Error processing {filename}: {e}")
                results.append({
                    'filename': filename,
                    'sample_type': 'error'
                })

    # 写入 CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'sample_type']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


# 使用示例
labels_folder = '/home/csz_changsha/data/yanqing_data/stitched_tiny_2/0422/labels'  # 你的labels文件夹路径
output_csv = '/home/csz_changsha/data/yanqing_data/stitched_tiny_2/0422/sample_classification.csv'  # 输出的CSV文件名
classify_samples_from_txt(labels_folder, output_csv)
# classify_samples_from_json(labels_folder, output_csv)
