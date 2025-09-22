# conda activate dogaction

import os
import torch
import cv2
import numpy as np

# 加载预训练的YOLOv5模型（yolov5s为小型模型）
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s')  # 这里加载YOLOv5的预训练模型

# 设置输入和输出根目录路径
input_root_dir = '/mnt/pami26/zengyi/backup/zengyi/doge/data/Doge_dataset'
output_root_dir = '/mnt/pami26/zengyi/backup/zengyi/doge/data/refined_Doge_dataset'  # 输出文件夹路径

# 遍历目录并处理每个子目录中的视频文件
for subdir, dirs, files in os.walk(input_root_dir):
    for file in files:
        if file.endswith(".mp4"):  # 只处理视频文件
            input_video_path = os.path.join(subdir, file)
            # 根据输入视频的路径构造输出路径
            relative_path = os.path.relpath(input_video_path, input_root_dir)
            output_video_path = os.path.join(output_root_dir, relative_path)

            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

            # 打开视频文件
            cap = cv2.VideoCapture(input_video_path)

            # 获取视频的基本信息：帧率、宽度、高度
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 设置输出视频的编码格式和大小（输出视频大小一致）
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 设置编码格式为mp4
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

            # 第一个循环：遍历所有帧，计算最小包围框的并集
            x_min, y_min, x_max, y_max = width, height, 0, 0
            valid_frame_count = 0  # 统计有效帧的数量

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 使用YOLO模型对当前帧进行检测
                results = model(frame)  # 将图像传递给YOLO模型进行推理
                boxes = results.xyxy[0].cpu().numpy()  # 获取检测框的坐标

                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box  # 解包检测框的坐标和信息
                    if int(cls) == 16:  # 如果是狗的类别（根据COCO数据集，狗的类别ID是16）
                        x_min = min(x_min, x1)
                        y_min = min(y_min, y1)
                        x_max = max(x_max, x2)
                        y_max = max(y_max, y2)

                valid_frame_count += 1

            # 如果没有有效的帧（没有检测到任何狗），则不进行裁剪
            if valid_frame_count == 0:
                print(f"没有有效的狗检测框，跳过文件: {input_video_path}")
                cap.release()
                out.release()
                continue

            # 计算最终裁剪区域并加上较大的padding
            padding = 100  # 设置padding大小
            width_ratio = float(width) / height  # 计算视频的宽高比
            height_ratio = float(height) / width

            # 计算裁剪区域的长宽比保持一致
            box_width = x_max - x_min
            box_height = y_max - y_min
            box_aspect_ratio = box_width / box_height

            # 让裁剪框的长宽比大体保持和视频一致
            if box_aspect_ratio > width_ratio:  # 如果检测框的长宽比大于视频的宽高比
                new_width = int((y_max - y_min) * width_ratio)
                new_height = y_max - y_min
            else:  # 如果检测框的长宽比小于或等于视频的宽高比
                new_height = int((x_max - x_min) * height_ratio)
                new_width = x_max - x_min

            # 计算裁剪区域的中心
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            # 调整裁剪框的大小，确保不超出视频边界
            x1 = max(0, int(center_x - new_width / 2) - padding)
            y1 = max(0, int(center_y - new_height / 2) - padding)
            x2 = min(width, int(center_x + new_width / 2) + padding)
            y2 = min(height, int(center_y + new_height / 2) + padding)

            print(f"最终裁剪区域坐标: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

            # 第二个循环：遍历所有帧，按最终的裁剪框对每一帧做相同位置的裁剪
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置帧读取位置
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 裁剪区域
                dog_crop = frame[y1:y2, x1:x2]

                # 如果裁剪的区域为空，跳过此帧
                if dog_crop.size == 0:
                    print(f"警告：裁剪区域为空，跳过此帧")
                    continue

                # 调整裁剪后的图像为原始视频大小
                dog_crop_resized = cv2.resize(dog_crop, (width, height))

                # 将调整后的狗区域帧写入输出视频
                out.write(dog_crop_resized)

            # 释放资源
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            print(f"处理完成，输出视频已保存为 {output_video_path}")