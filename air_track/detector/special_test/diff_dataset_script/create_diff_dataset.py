#!/usr/bin/env python3
"""Create dataset difference C = A - B and relabel all C labels to point class."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create C from dataset A and B where C = A - B (by relative file paths), "
            "keeping directory layout and relabeling labels to point class."
        )
    )
    parser.add_argument("--a", required=True, type=Path, help="Path of dataset A")
    parser.add_argument("--b", required=True, type=Path, help="Path of dataset B")
    parser.add_argument("--c", required=True, type=Path, help="Output path for dataset C")
    parser.add_argument(
        "--classes-src",
        type=Path,
        default=None,
        help=(
            "Path to classes.txt to copy into every labels/ directory in C. "
            "If not set, the script auto-detects classes.txt in A."
        ),
    )
    parser.add_argument(
        "--point-class",
        default="0",
        help="Class token used for point label in YOLO txt (default: 0)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty C directory",
    )
    parser.add_argument(
        "--no-pair-sync",
        action="store_true",
        help=(
            "Disable image-label pair sync. By default, when an image is in diff, "
            "its matching label from A is also copied (if exists)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview counts only; no files are copied/modified",
    )
    return parser.parse_args()


def list_relative_files(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
    }


def is_label_file(relative_path: str) -> bool:
    rel = Path(relative_path)
    return rel.suffix == ".txt" and "labels" in rel.parts


def is_image_file(relative_path: str) -> bool:
    return Path(relative_path).suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    } and "images" in Path(relative_path).parts


def image_to_label_rel(relative_path: str) -> str | None:
    rel = Path(relative_path)
    parts = list(rel.parts)
    if "images" not in parts:
        return None
    idx = parts.index("images")
    parts[idx] = "labels"
    return str(Path(*parts).with_suffix(".txt").as_posix())


def expand_with_pairs(diff_files: set[str], files_a: set[str]) -> tuple[set[str], int]:
    extras: set[str] = set()
    for rel in diff_files:
        if is_image_file(rel):
            label_rel = image_to_label_rel(rel)
            if label_rel and label_rel in files_a:
                extras.add(label_rel)
    return diff_files | extras, len(extras)


def find_classes_source(path_a: Path, classes_src_arg: Path | None) -> Path | None:
    if classes_src_arg is not None:
        src = classes_src_arg.resolve()
        return src if src.is_file() else None

    root_candidate = path_a / "classes.txt"
    if root_candidate.is_file():
        return root_candidate

    candidates = sorted(path_a.rglob("classes.txt"))
    return candidates[0] if candidates else None


def copy_classes_to_labels(path_c: Path, classes_src: Path) -> int:
    label_dirs = sorted(p for p in path_c.rglob("labels") if p.is_dir())
    for label_dir in label_dirs:
        shutil.copy2(classes_src, label_dir / "classes.txt")
    return len(label_dirs)


def rewrite_label_file(label_path: Path, point_class: str) -> bool:
    """Rewrite class token to point_class for every non-empty line.

    Returns True if file content changed.
    """
    original = label_path.read_text(encoding="utf-8")
    lines = original.splitlines()
    changed = False
    output_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            output_lines.append(line)
            continue

        tokens = stripped.split()
        if not tokens:
            output_lines.append(line)
            continue

        if tokens[0] != point_class:
            changed = True
            tokens[0] = point_class

        output_lines.append(" ".join(tokens))

    new_text = "\n".join(output_lines)
    if original.endswith("\n"):
        new_text += "\n"

    if changed:
        label_path.write_text(new_text, encoding="utf-8")

    return changed


def main() -> int:
    args = parse_args()

    path_a = args.a.resolve()
    path_b = args.b.resolve()
    path_c = args.c.resolve()

    if not path_a.is_dir():
        print(f"[ERROR] A does not exist or is not a directory: {path_a}")
        return 1
    if not path_b.is_dir():
        print(f"[ERROR] B does not exist or is not a directory: {path_b}")
        return 1
    if path_c == path_a or path_c == path_b:
        print("[ERROR] C path must be different from A and B")
        return 1

    if path_c.exists() and any(path_c.iterdir()) and not args.overwrite:
        print(f"[ERROR] C already exists and is not empty: {path_c}")
        print("        Use --overwrite if you want to write into it.")
        return 1

    files_a = list_relative_files(path_a)
    files_b = list_relative_files(path_b)
    diff_set = files_a - files_b
    paired_added = 0
    if not args.no_pair_sync:
        diff_set, paired_added = expand_with_pairs(diff_set, files_a)
    diff_files = sorted(diff_set)

    print(f"A files: {len(files_a)}")
    print(f"B files: {len(files_b)}")
    print(f"C = A - B files to copy: {len(diff_files)}")
    if not args.no_pair_sync:
        print(f"Pair-sync label files added from A: {paired_added}")

    classes_src = find_classes_source(path_a, args.classes_src)
    if classes_src is None:
        print("[WARN] classes.txt not found; skip copying classes.txt into labels/")
    else:
        print(f"classes.txt source: {classes_src}")

    if args.dry_run:
        label_count = sum(1 for rel in diff_files if is_label_file(rel))
        print(f"[DRY-RUN] label files in C to relabel: {label_count}")
        if classes_src is not None:
            print("[DRY-RUN] classes.txt will be copied to all labels/ directories in C")
        return 0

    path_c.mkdir(parents=True, exist_ok=True)

    copied = 0
    for rel in diff_files:
        src = path_a / rel
        dst = path_c / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    label_files = [
        p for p in path_c.rglob("*.txt") if p.is_file() and "labels" in p.parts
    ]
    changed_labels = 0
    for label_file in label_files:
        if rewrite_label_file(label_file, args.point_class):
            changed_labels += 1

    classes_copied = 0
    if classes_src is not None:
        classes_copied = copy_classes_to_labels(path_c, classes_src)

    print(f"Copied files to C: {copied}")
    print(f"Label files scanned in C: {len(label_files)}")
    print(f"Label files changed to point({args.point_class}): {changed_labels}")
    print(f"classes.txt copied to labels/ dirs: {classes_copied}")
    print(f"Done. Output C: {path_c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
