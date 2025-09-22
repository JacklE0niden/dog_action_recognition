import os

def rename_files(src_dir):
    # 定义动作类别
    categories = ["stand", "sit", "lay", "sniff", "walk", "run", "jump", "roll", "pounce"]
    
    # 用于保存每个类别出现的计数
    category_counters = {category: 1 for category in categories}

    # 遍历目录中的所有文件
    for file in os.listdir(src_dir):
        # 检查是否是 JSON 文件
        if file.endswith(".json"):
            # 获取文件的前缀（动作类别）
            for category in categories:
                if file.lower().startswith(category):
                    # 获取当前类别的计数
                    counter = category_counters[category]
                    
                    # 构造新的文件名，例如 walk01.json, walk02.json 等
                    new_name = f"{category}{counter:02d}.json"
                    
                    # 更新该类别的计数器
                    category_counters[category] += 1
                    
                    # 获取原文件和新文件的完整路径
                    old_file_path = os.path.join(src_dir, file)
                    new_file_path = os.path.join(src_dir, new_name)
                    
                    # 重命名文件
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: {old_file_path} -> {new_file_path}")
                    break  # 如果已经匹配到一个类别，跳出循环

if __name__ == "__main__":
    src_dir = "datasets/train"  # 目标目录，包含需要重命名的 JSON 文件
    rename_files(src_dir)