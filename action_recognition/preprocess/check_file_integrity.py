import os
import subprocess

def repair_mp4(input_path, output_path):
    """
    尝试使用 ffmpeg 修复 mp4 文件，保留音视频流，并重新封装。
    """
    command = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-err_detect', 'ignore_err',
        '-i', input_path,
        '-c', 'copy',
        output_path
    ]

    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    if result.returncode != 0 or b"Error" in result.stderr or b"partial file" in result.stderr:
        return False
    return True

def repair_files_in_directory(root_dir, repaired_dir):
    """
    遍历根目录下的所有子目录，修复所有 mp4 文件
    """
    if not os.path.exists(repaired_dir):
        os.makedirs(repaired_dir)

    error_log = []  # 记录未修复的文件

    # 遍历目录及其子目录
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith('.mp4'):
                # 获取文件的完整路径
                filepath = os.path.join(root, filename)
                # 在修复目录中创建对应的子目录
                relative_path = os.path.relpath(filepath, root_dir)
                output_path = os.path.join(repaired_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                print(f"⏳ 正在修复: {relative_path}")
                # 修复并检查
                success = repair_mp4(filepath, output_path)

                if success:
                    print(f"✅ 修复成功: {relative_path}")
                else:
                    print(f"❌ 修复失败: {relative_path}")
                    error_log.append(relative_path)

    # 保存失败日志
    with open("repair_failed.txt", "w") as f:
        for item in error_log:
            f.write(item + "\n")

    print(f"\n处理完成，总计失败 {len(error_log)} 个文件，详情见 repair_failed.txt")

# 调用函数
source_dir = "/mnt/pami26/zengyi/backup/zengyi/doge/data/refined_Doge_dataset_unintegral"  # 你要修复的目录路径
repaired_dir = "/mnt/pami26/zengyi/backup/zengyi/doge/data/refined_Doge_dataset"  # 修复后保存的目录路径

repair_files_in_directory(source_dir, repaired_dir)
