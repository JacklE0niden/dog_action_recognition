import json
import os

def convert_keypoints_format(input_dir, output_dir, label_file):
    # 读取标签文件
    with open(label_file, 'r') as label_file:
        labels = json.load(label_file)
    
    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        # 仅处理 JSON 文件
        if filename.endswith('.json'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename)
            
            with open(input_file, 'r') as file:
                data = json.load(file)
            
            # 获取文件名作为键来查找标签
            filenamekey = os.path.splitext(filename)[0]
            if filenamekey in labels:
                label = labels[filenamekey]['label']
                label_index = labels[filenamekey]['label_index']
                print(f"Processing file: {filename}, Label: {label}, Label Index: {label_index}")
            else:
                print(f"Warning: Label for {filenamekey} not found in label file!")
                continue

            converted_data = {"data": [], "label": label, "label_index": label_index}
            
            for frame in data['frames']:  # 对每一帧的循环
                frame_index = frame['frame_id']
                skeleton = []
                
                for dog in frame['dogs']:
                    if dog['keypoints']:
                        pose = []
                        score = []
                        for keypoint in dog['keypoints'][0]:  # 只取前两个关键点
                            pose.extend([round(coord, 3) for coord in keypoint[:2]])  # 保留前两个坐标三位小数
                            score.append(round(keypoint[2], 3))  # 保留置信度三位小数
                        skeleton.append({"pose": pose, "score": score})
                
                # 如果没有检测到狗，则将 skeleton 设置为空列表
                if not pose:
                    converted_data["data"].append({"frame_index": frame_index, "skeleton": []})
                else:
                    converted_data["data"].append({"frame_index": frame_index, "skeleton": skeleton})
            
            # 保存转换后的数据到输出目录
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, 'w') as outfile:
                json.dump(converted_data, outfile, separators=(', ', ': '), ensure_ascii=False)
            print(f"Converted file saved as {output_file}")

# 调用函数并指定输入文件夹、输出文件夹和标签文件路径
input_dir = 'datasets/train'  # 输入文件夹，包含原始的 JSON 文件
output_dir = 'datasets/dog_kinetics/dog_kinetics_train'  # 输出文件夹，保存转换后的 JSON 文件
label_file = 'datasets/dog_kinetics/dog_kinetics_train_label.json'  # 标签文件路径

convert_keypoints_format(input_dir, output_dir, label_file)


# 调用函数并指定输入文件夹、输出文件夹和标签文件路径
input_dir2 = 'datasets/val'  # 输入文件夹，包含原始的 JSON 文件
output_dir2 = 'datasets/dog_kinetics/dog_kinetics_val'  # 输出文件夹，保存转换后的 JSON 文件
label_file2 = 'datasets/dog_kinetics/dog_kinetics_val_label.json'  # 标签文件路径

convert_keypoints_format(input_dir2, output_dir2, label_file2)