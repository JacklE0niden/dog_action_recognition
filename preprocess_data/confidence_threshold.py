import json
import os

def update_keypoints_confidence_in_directory(directory_path):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        # 只处理以 '.json' 结尾的文件
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            print("Processing file:", file_path)

            # 检查文件是否为空
            if os.path.getsize(file_path) == 0:
                print(f"Skipping empty file: {file_path}")
                continue
            
            try:
                # 尝试打开并解析 JSON 文件
                with open(file_path, 'r') as file:
                    data = json.load(file)

                # 检查文件结构是否符合预期
                if 'frames' not in data:
                    print(f"Invalid structure in {file_path}: missing 'frames' key")
                    continue

                # 遍历每一帧数据
                for frame in data['frames']:
                    if 'dogs' not in frame:
                        print(f"Missing 'dogs' in frame in {file_path}")
                        continue
                    
                    # 遍历每个狗的数据
                    for dog in frame['dogs']:
                        if 'keypoints' not in dog:
                            print(f"Missing 'keypoints' in dog data in {file_path}")
                            continue
                        
                        # 遍历每个 keypoints 列表
                        for keypoint_set in dog['keypoints']:
                            for i, keypoint in enumerate(keypoint_set):
                                # 检查 keypoint 是否有足够的元素
                                if len(keypoint) >= 3:
                                    x, y, confidence = keypoint[0], keypoint[1], keypoint[2]
                                    # 检查 x, y 坐标是否为 0.0 且置信度小于 0.5
                                    if x == 0.0 and y == 0.0 and confidence < 0.5:
                                        keypoint_set[i][2] = 0.0  # 将置信度设为 0.0
                                else:
                                    print(f"Skipping invalid keypoint in {file_path}, frame {frame.get('frame_id', 'unknown')}, dog {dog.get('dog_id', 'unknown')}")

                # 将更新后的数据保存回文件
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)
                print(f"Updated file: {file_path}")

            except json.JSONDecodeError:
                print(f"Error decoding JSON in {file_path}. Skipping file.")
            except Exception as e:
                print(f"Unexpected error with {file_path}: {e}")

# 使用示例
update_keypoints_confidence_in_directory('datasets/train')