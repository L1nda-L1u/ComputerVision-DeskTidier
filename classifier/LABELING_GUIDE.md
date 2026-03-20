# 打标签指南 (Tidy / Untidy)

## 方式一：键盘标注脚本（推荐）

```bash
pip install opencv-python
python label_desk.py
```

- 逐张显示图片
- 按 **1** 或 **t** = tidy（整洁）
- 按 **2** 或 **u** = untidy（凌乱）
- 按 **s** = 跳过
- 按 **q** = 保存并退出

标签会保存到 `desk_labels.csv`，可随时继续标注（已标过的会跳过）。

---

## 方式二：手动建文件夹

1. 创建两个文件夹：
   ```
   dataset/
     tidy/
     untidy/
   ```
2. 把图片拖到对应文件夹里

---

## 标注完成后

运行 `prepare_dataset.py` 会根据 `desk_labels.csv` 生成 `dataset/tidy/` 和 `dataset/untidy/`：

```bash
python prepare_dataset.py
```

之后用 ResNet18 做 transfer learning 时，直接读取 `dataset/` 即可。
