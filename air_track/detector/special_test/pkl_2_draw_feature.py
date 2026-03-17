import glob
import os
import pickle
from air_track.detector.visualization.visualize_and_save import draw_feature_img_orig_data


def pkl_2_draw_feature(pkl_path, save_path, img_end='.pkl'):
    """找到指定文件夹下所有的pkl文件，将保存的pkl用plt画图"""
    pkl_files = glob.glob(os.path.join(pkl_path, f'*{img_end}'))
    for file in pkl_files:
        name = os.path.basename(file)
        with open(file, 'rb') as f:
            new_name = name.replace(f'{img_end}', '.png')
            data = pickle.load(f)
            draw_feature_img_orig_data(data, save_path + f'/model_pred_without_star/{new_name}', is_plt_star=False)


if __name__ == "__main__":
    """保存的pkl可视化"""
    pkl_path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/visual_feature_changed_labels_test/model_pred_data'
    save_path = '/home/csz_changsha/PycharmProjects/github/zhuang5252/AirTrack/visual_feature_changed_labels_test/model_pred_without_star_1'

    pkl_2_draw_feature(pkl_path, save_path, img_end='.pkl')
    print('Draw feature successful')
