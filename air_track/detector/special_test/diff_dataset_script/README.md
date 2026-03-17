# 数据差集脚本说明

脚本：`create_diff_dataset.py`

功能：
- 按相对路径计算差集：`C = A - B`
- 保持目录结构复制到新目录 `C`
- 自动补齐差集中图片对应的标签（可关闭）
- 将 `C` 中 `labels/*.txt` 的类别统一改为 `point`（默认类别 id 为 `0`）
- 将 `classes.txt` 复制到 `C` 下所有 `labels/` 目录

## 常用参数

- `--a`：数据集 A 路径
- `--b`：数据集 B 路径
- `--c`：输出数据集 C 路径
- `--point-class`：`point` 对应类别（默认 `0`）
- `--classes-src`：指定 `classes.txt` 路径（不传则在 A 中自动查找）
- `--no-pair-sync`：关闭“图片差集自动补齐对应标签”
- `--overwrite`：允许写入已存在且非空的 C
- `--dry-run`：只统计不落盘

## 示例

```bash
python3 create_diff_dataset.py \
  --a /path/to/A \
  --b /path/to/B \
  --c /path/to/C
```


