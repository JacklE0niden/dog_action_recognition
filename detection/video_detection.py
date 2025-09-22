import cv2
import numpy as np
import torch

# 加载 YOLOv5 模型（YOLOv5s 作为示例）
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 可以使用更大版本如 'yolov5m' 或 'yolov5l'
# 读取视频
video_path = 'your_video.mp4'
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率（FPS）
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 用于保存结果
output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换 BGR 到 RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 使用 YOLO 模型进行推理
    results = model(img_rgb)

    # 获取检测到的物体类别与坐标
    # 返回结果中的所有检测框
    boxes = results.xyxy[0].numpy()  # [xmin, ymin, xmax, ymax, confidence, class]
    
    # 过滤掉非狗的类别
    dog_class_id = 16  # 16 对应的是 COCO 数据集中“狗”的类别
    dog_boxes = [box for box in boxes if box[5] == dog_class_id]

    # 创建一个 mask，标记狗的位置
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    for box in dog_boxes:
        xmin, ymin, xmax, ymax = map(int, box[:4])
        mask[ymin:ymax, xmin:xmax] = 255  # 在狗的区域设置为白色

    # 将狗区域抠出来
    dog_region = cv2.bitwise_and(frame, frame, mask=mask)

    # 写入处理后的帧
    out.write(dog_region)

    # 如果需要在帧上显示，可以取消注释
    # cv2.imshow("Frame", dog_region)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()