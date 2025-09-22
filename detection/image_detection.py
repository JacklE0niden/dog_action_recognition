from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import argparse
from tqdm import tqdm  # 添加进度条显示

# 保留周围
# def detect_dog(image_path, output_dir):
#     # 加载YOLOv8模型
#     model = YOLO('yolov8n.pt')
    
#     # 读取图片
#     img = cv2.imread(str(image_path))
#     if img is None:
#         print(f"错误：无法读取图片 {image_path}")
#         return False
    
#     # 进行检测
#     results = model(img, classes=[16])  # 16是狗的类别ID
    
#     # 在图片上绘制检测框
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # 获取边界框坐标
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             # 获取置信度
#             conf = float(box.conf[0])
            
#             # 绘制边界框
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             # 添加置信度标签
#             label = f'Dog {conf:.2f}'
#             cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#     # 生成输出文件名
#     output_path = output_dir / f'{image_path.stem}_detected{image_path.suffix}'
    
#     # 保存结果图片
#     cv2.imwrite(str(output_path), img)
#     return True

# 去掉周围
def detect_dog(image_path, output_dir):
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"错误：无法读取图片 {image_path}")
        return False
    
    detected_regions = []
    # 进行检测
    results = model(img, classes=[16])  # 16是狗的类别ID
    
    # 处理检测到的每个边界框
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 获取边界框坐标和置信度
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # 只处理置信度大于0.4的检测结果
            if conf > 0.4:
                # 裁剪出检测区域
                cropped_img = img[y1:y2, x1:x2]
                detected_regions.append((cropped_img, conf))
    
    # 如果检测到狗，保存裁剪的图像
    if detected_regions:
        for i, (cropped_img, conf) in enumerate(detected_regions):
            # 为多个检测结果生成不同的文件名
            output_path = output_dir / f'{image_path.stem}_dog_{i+1}_{conf:.2f}{image_path.suffix}'
            cv2.imwrite(str(output_path), cropped_img)
        return True
    return False


def process_directory(input_dir, output_dir):
    # 确保输入目录存在
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误：输入目录 {input_dir} 不存在")
        return
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # 获取所有图片文件
    image_files = [f for f in input_path.rglob('*') if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"警告：在 {input_dir} 中没有找到支持的图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 处理每个图片
    successful = 0
    failed = 0
    
    for image_file in tqdm(image_files, desc="处理图片"):
        if detect_dog(image_file, output_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\n处理完成：")
    print(f"成功：{successful} 张图片")
    print(f"失败：{failed} 张图片")
    print(f"处理结果保存在：{output_path}")



if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='使用YOLOv8检测图片中的狗')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图片目录的路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录的路径')
    
    # 解析参数
    args = parser.parse_args()
    
    # 加载YOLOv8模型（全局变量，避免重复加载）
    model = YOLO('yolov8n.pt')
    
    # 处理目录
    process_directory(args.input_dir, args.output_dir)
