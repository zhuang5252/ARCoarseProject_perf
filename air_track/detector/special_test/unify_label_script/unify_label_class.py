#!/usr/bin/env python3
"""Unify YOLO detection label class IDs under one target class.

Example:
  python3 unify_label_class.py --labels-dir labels --class-id 0
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将目标文件夹下所有标签文件的类别ID统一为同一个值，"
            "并跳过 classes.txt。"
        )
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        type=Path,
        help="标签目录，例如: labels",
    )
    parser.add_argument(
        "--class-id",
        required=True,
        type=int,
        help="目标类别ID（非负整数），例如: 0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅预览将要修改的文件数量，不写回文件。",
    )
    return parser.parse_args()


def update_label_file(file_path: Path, class_id: int, dry_run: bool) -> tuple[bool, int]:
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    new_lines: list[str] = []
    changed_lines = 0

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            new_lines.append(raw_line)
            continue

        parts = stripped.split()
        if not parts:
            new_lines.append(raw_line)
            continue

        new_line = " ".join([str(class_id), *parts[1:]])
        if new_line != stripped:
            changed_lines += 1
        new_lines.append(new_line)

    changed = changed_lines > 0
    if changed and not dry_run:
        output = "\n".join(new_lines)
        if content.endswith("\n"):
            output += "\n"
        file_path.write_text(output, encoding="utf-8")

    return changed, changed_lines


def main() -> int:
    args = parse_args()

    if args.class_id < 0:
        raise SystemExit("--class-id 必须是非负整数。")

    labels_dir: Path = args.labels_dir
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise SystemExit(f"标签目录不存在或不是目录: {labels_dir}")

    label_files = sorted(
        p for p in labels_dir.glob("*.txt") if p.name.lower() != "classes.txt"
    )

    if not label_files:
        print("未找到可处理的标签文件（已自动跳过 classes.txt）。")
        return 0

    changed_files = 0
    changed_total_lines = 0

    for file_path in label_files:
        changed, changed_lines = update_label_file(file_path, args.class_id, args.dry_run)
        if changed:
            changed_files += 1
            changed_total_lines += changed_lines

    mode = "预览模式" if args.dry_run else "写入模式"
    print(
        f"{mode}: 共扫描 {len(label_files)} 个文件，"
        f"修改 {changed_files} 个文件，替换 {changed_total_lines} 行类别ID。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
