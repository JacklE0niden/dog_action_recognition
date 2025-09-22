import os
import shutil

def copy_json_files(src_dir, dest_dir):
    # 如果目标目录不存在，则创建
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 遍历所有子目录
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # 只复制 JSON 文件
            if file.endswith(".json"):
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir, file)
                
                # 如果文件已存在，添加后缀
                counter = 1
                while os.path.exists(dest_file):
                    # 如果目标文件已经存在，给文件名加后缀
                    name, ext = os.path.splitext(file)
                    dest_file = os.path.join(dest_dir, f"{name}_{counter}{ext}")
                    counter += 1

                # 复制文件
                shutil.copy(src_file, dest_file)
                print(f"Copied {src_file} to {dest_file}")

if __name__ == "__main__":
    src_dir = "datasets/train_videos"  # 源目录
    dest_dir = "datasets/train"  # 目标目录

    copy_json_files(src_dir, dest_dir)