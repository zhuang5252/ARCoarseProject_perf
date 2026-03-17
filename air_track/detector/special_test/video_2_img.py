import cv2
import os


def video_to_frames(video_path, output_dir, frame_prefix='frame_', image_format='jpg',
                    start_frame=0, end_frame=None, interval=1):
    """
    将视频拆分为帧图像（支持抽帧）

    参数：
    video_path: 输入视频文件路径（如：'input.mp4'）
    output_dir: 输出目录路径（如：'output_frames'）
    frame_prefix: 帧文件名前缀（默认：'frame_'）
    image_format: 图像格式（支持jpg/png等）
    start_frame: 起始帧序号（默认从0开始）
    end_frame: 结束帧序号（默认到视频结尾）
    interval: 抽帧间隔（默认1即不抽帧，2表示每隔1帧抽1帧）
    """
    # 参数校验
    if interval < 1:
        raise ValueError("抽帧间隔必须大于等于1")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件，请检查路径是否有效")

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame or total_frames, total_frames)

    print(f"总帧数：{total_frames}，抽帧间隔：{interval}，预计输出：{(end_frame - start_frame) // interval} 帧")

    # 跳转到起始帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    saved_count = 0

    # 抽帧循环
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # 保存当前帧
        frame_number = str(saved_count).zfill(6)  # 按保存顺序编号
        output_path = os.path.join(output_dir, f"{frame_prefix}{frame_number}.{image_format}")
        cv2.imwrite(output_path, frame)
        saved_count += 1

        # 显示进度
        if saved_count % 100 == 0:
            print(f"已保存 {saved_count} 帧，当前处理到第 {current_frame} 帧...")

        # 跳过指定间隔的帧
        skip_frames = interval - 1
        while skip_frames > 0 and current_frame <= end_frame:
            cap.grab()  # 快速跳帧（不解码）
            current_frame += 1
            skip_frames -= 1

        current_frame += 1

    # 释放资源
    cap.release()
    print(f"处理完成！实际保存 {saved_count} 帧到 {output_dir}")


# 使用示例
if __name__ == "__main__":
    video_to_frames(
        video_path="/home/csz_changsha/data/yanqing_data/205mp4/1true_fsc_1false_fsc_infrared.mp4",
        output_dir="/home/csz_changsha/data/yanqing_data/205mp4/1true_fsc_1false_fsc_infrared",
        image_format="jpg",
        start_frame=0,
        interval=50,
        # end_frame=3000  # 可选：只处理前300帧
    )
