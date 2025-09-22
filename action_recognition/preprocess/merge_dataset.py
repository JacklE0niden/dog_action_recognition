# conda activate dogvideo
'''
    把两个数据集合并，要求两个数据集都是无子文件夹的，并且已经命名过类别标签
'''

import os
import shutil
import re
from collections import defaultdict

def get_category_and_index(filename):
    match = re.match(r"([a-zA-Z]+)(\d+)\.mp4", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def merge_and_rename(folder1, folder2, output_root):
    os.makedirs(output_root, exist_ok=True)
    category_counts = defaultdict(int)

    # Step 1: 统计 folder1 中每个类别已有的数量，并复制到输出目录
    for filename in sorted(os.listdir(folder1)):
        if filename.endswith(".mp4"):
            category, _ = get_category_and_index(filename)
            if category:
                category_dir = os.path.join(output_root, category)
                os.makedirs(category_dir, exist_ok=True)

                category_counts[category] += 1
                new_name = f"{category}{category_counts[category]:02d}.mp4"
                shutil.copy(
                    os.path.join(folder1, filename),
                    os.path.join(category_dir, new_name)
                )

    # Step 2: 将 folder2 中的文件继续命名并复制
    for filename in sorted(os.listdir(folder2)):
        if filename.endswith(".mp4"):
            category, _ = get_category_and_index(filename)
            if category:
                category_dir = os.path.join(output_root, category)
                os.makedirs(category_dir, exist_ok=True)

                category_counts[category] += 1
                new_name = f"{category}{category_counts[category]:02d}.mp4"
                shutil.copy(
                    os.path.join(folder2, filename),
                    os.path.join(category_dir, new_name)
                )

    print("合并完成 ✅")

if __name__ == "__main__":
    folder1 = "/mnt/pami23/zengyi/DogCentric/extra_dog_video/collect"
    folder2 = "zengyi/DogCentric/extra_dog_video/web"
    output_root = "zengyi/DogCentric/extra_dog_video/Doge_dataset"

    merge_and_rename(folder1, folder2, output_root)