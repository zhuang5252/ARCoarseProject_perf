# 标签类别统一脚本

用于将目标检测标签文件中的类别 ID 统一改成同一个值。  
`classes.txt` 不会被修改。

## 文件

- `unify_label_class.py`：批量改写标签类别 ID 的脚本

## 用法

在当前目录执行：

```bash
# 1) 预览将修改多少文件（不写入）
python3 unify_label_class.py --labels-dir labels --class-id 0 --dry-run

# 2) 正式写入
python3 unify_label_class.py --labels-dir labels --class-id 0
```

## 参数说明

- `--labels-dir`：标签目录（如 `labels`）
- `--class-id`：目标类别 ID（非负整数）
- `--dry-run`：仅预览，不落盘

## 说明

- 脚本会处理 `labels` 目录下所有 `.txt` 标签文件
- 自动跳过 `classes.txt`
- 每行仅替换首列类别 ID，后面的坐标值保持不变
