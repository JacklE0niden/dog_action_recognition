import os

# 设置文件夹路径
folder_path = '/mnt/pami26/zengyi/dog_action/datasets/dog_kinetics_xpose/dog_kinetics_train'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('_interpolated.json'):
        # 生成新文件名
        new_filename = filename.replace('_interpolated', '')
        
        # 构造完整的文件路径
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        
        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')