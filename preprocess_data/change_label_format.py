# 原始格式
# {
#     "categories": [
#         "stand",
#         "sit",
#         "lay",
#         "sniff",
#         "walk",
#         "run",
#         "jump",
#         "roll",
#         "pounce"
#     ],
#     "annotations": {
#         "stand01.json": {
#             "category_id": 1
#         },
#         "stand02.json": {
#             "category_id": 1
#         },
#         "stand03.json": {
#             "category_id": 1
#         }
#     }
# }
# 目标格式
# {
#     "--07WQ2iBlw": {
#         "has_skeleton": true, 
#         "label": "javelin throw", 
#         "label_index": 166
#     }, 
#     "--33Lscn6sk": {
#         "has_skeleton": true, 
#         "label": "flipping pancake", 
#         "label_index": 129
#     }
# }
import json

def convert_annotation(input_json_path, output_json_path):
    # 读取原始 JSON 文件
    with open(input_json_path, 'r') as file:
        data = json.load(file)

    # 创建新的格式
    new_format = {}
    categories = data["categories"]
    print("categories:", categories)

    # 遍历每个视频的注释
    for video_file, annotation in data['annotations'].items():
        # 获取文件名（去除扩展名）
        file_name = video_file.split('.')[0]
        print("file_name:", file_name)
        
        # 获取 category_id
        category_id = annotation.get('category_id')
        print("category_id:", category_id)
        
        # 生成新的格式
        new_format[file_name] = {
            "has_skeleton": True,
            "label": categories[category_id - 1],  # 索引调整为从 0 开始
            "label_index": category_id
        }

    # 将新格式写入输出文件
    with open(output_json_path, 'w') as outfile:
        json.dump(new_format, outfile, indent=4)

    print(f"转换完成，结果已保存到 {output_json_path}")




# 调用函数处理 train 和 val 注释文件
train_input_json = 'datasets/train_annotation.json'  # 替换为实际文件路径
train_output_json = 'datasets/dog_kinetics/dog_kinetics_train_label.json'
convert_annotation(train_input_json, train_output_json)

val_input_json = 'datasets/val_annotation.json'  # 替换为实际文件路径
val_output_json = 'datasets/dog_kinetics/dog_kinetics_val_label.json'
convert_annotation(val_input_json, val_output_json)