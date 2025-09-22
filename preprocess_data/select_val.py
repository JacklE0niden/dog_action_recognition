# 在datasets/dog_kinetics/train文件夹中有很多个json文件
# 你需要每隔5个文件选一个作为测试集，复制存放到datasets/dog_kinetics/val中
import os
import shutil
import json

def select_test_files(src_dir, dest_dir, label_file, dest_label_file, interval=5):
    # 如果目标目录不存在，则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 获取所有 JSON 文件
    files = [file for file in os.listdir(src_dir) if file.endswith('.json')]

    # 按文件名排序（如果有必要的话）
    files.sort()

    # 读取标签文件
    with open(label_file, 'r') as f:
        labels = json.load(f)

    selected_labels = {}

    # 每隔 interval 个文件选择一个文件并复制到目标目录
    for i, file in enumerate(files):
        if i % interval == 0:  # 每隔 interval 个文件选择一个
            src_file = os.path.join(src_dir, file)
            dest_file = os.path.join(dest_dir, file)

            # 复制文件
            shutil.copy(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")

            # 选择相应的标签
            filenamekey = file.split('.')[0]  # 获取文件名（不带扩展名）
            if filenamekey in labels:
                selected_labels[filenamekey] = labels[filenamekey]

    # 将选择的标签写入新的标签文件
    with open(dest_label_file, 'w') as f:
        json.dump(selected_labels, f, ensure_ascii=False, indent=4)
    print(f"Labels saved to {dest_label_file}")

if __name__ == "__main__":
    src_dir = 'datasets/dog_kinetics/dog_kinetics_train'  # 输入目录
    dest_dir = 'datasets/dog_kinetics/dog_kinetics_val'   # 目标目录
    label_file = 'datasets/dog_kinetics/dog_kinetics_train_label.json'  # 标签文件
    dest_label_file = 'datasets/dog_kinetics/dog_kinetics_val_label.json'  # 新标签文件

    select_test_files(src_dir, dest_dir, label_file, dest_label_file)