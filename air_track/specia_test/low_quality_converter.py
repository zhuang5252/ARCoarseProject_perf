# -*- coding: utf-8 -*-
# @Author    :
# @File      : AirTrack - low_quality_converter.py
# @Created   : 2025/09/13 22:26
# @Desc      : 压缩图片质量

"""
压缩图片质量
"""

import cv2
import os
import argparse

SUPPORT_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

def is_image_file(name: str) -> bool:
    return any(name.lower().endswith(ext) for ext in SUPPORT_EXTS)

def ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else max(1, n - 1)

def batch_resize(input_dir, output_dir, target_w=640, target_h=360,
                 enable_blur=False, blur_ksize=5,
                 enable_compress=False, jpeg_quality=30,
                 force_jpeg=False):
    if not os.path.isdir(input_dir):
        raise ValueError(f"输入目录不存在: {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    processed = 0
    for filename in os.listdir(input_dir):
        if not is_image_file(filename):
            continue

        in_path = os.path.join(input_dir, filename)
        img = cv2.imread(in_path)
        if img is None:
            print(f"⚠️ 无法读取图片: {in_path}")
            continue

        # 1) 缩放到固定尺寸
        resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # 2) 可选的高斯模糊（更接近“低清”观感）
        if enable_blur:
            k = ensure_odd(blur_ksize)
            resized = cv2.GaussianBlur(resized, (k, k), 0)

        # 3) 输出路径与扩展名
        name, ext = os.path.splitext(filename)
        if force_jpeg:
            out_name = f"{name}.jpg"
            out_path = os.path.join(output_dir, out_name)
            # 统一写成 JPEG；是否启用压缩由 jpeg_quality 控制
            cv2.imwrite(out_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        else:
            # 保持原扩展名不变
            out_path = os.path.join(output_dir, filename)

            # 仅当原本就是 jpg/jpeg 时，才应用 JPEG 压缩质量参数
            if enable_compress and ext.lower() in [".jpg", ".jpeg"]:
                cv2.imwrite(out_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            else:
                # 其他格式直接按默认参数保存（PNG/WEBP 等不会应用“JPEG 压缩”）
                cv2.imwrite(out_path, resized)

        processed += 1
        print(f"✅ 已处理: {filename}")

    print(f"\n完成！共处理 {processed} 张图片，已保存到: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量将图片缩放为 640x360，并可选高斯模糊与 JPEG 压缩")
    parser.add_argument("input", help="输入文件夹路径 (A)")
    parser.add_argument("output", help="输出文件夹路径 (B)")
    parser.add_argument("--width", type=int, default=640, help="目标宽度，默认 640")
    parser.add_argument("--height", type=int, default=360, help="目标高度，默认 360")

    # 模糊相关
    parser.add_argument("--blur", action="store_true", help="启用高斯模糊（更像低清）")
    parser.add_argument("--blur-ksize", type=int, default=3, help="高斯模糊核大小，需奇数，默认 3")

    # 压缩相关
    parser.add_argument("--compress", action="store_true", help="对 JPG/JPEG 启用 JPEG 压缩")
    parser.add_argument("--jpeg-quality", type=int, default=30, help="JPEG 压缩质量 (1-100, 越小越糊)，默认 30")
    parser.add_argument("--force-jpeg", action="store_true",
                        help="强制将所有输出转为 .jpg 并应用 JPEG 质量参数（覆盖原扩展名）")

    args = parser.parse_args()
    batch_resize(
        input_dir=args.input,
        output_dir=args.output,
        target_w=args.width,
        target_h=args.height,
        enable_blur=args.blur,
        blur_ksize=args.blur_ksize,
        enable_compress=args.compress,
        jpeg_quality=args.jpeg_quality,
        force_jpeg=args.force_jpeg
    )
