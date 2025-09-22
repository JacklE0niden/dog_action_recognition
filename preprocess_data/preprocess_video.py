import cv2
import os
import numpy as np
import argparse

def process_video(input_video_path, output_video_path, target_fps=30, target_frames=90):
    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    
    # 获取原始视频的帧数和帧率
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 打印原始视频信息
    print(f"Processing video: {os.path.basename(input_video_path)}")
    print(f"Original FPS: {original_fps}, Total Frames: {original_frame_count}")
    
    # 计算采样的帧间隔
    if original_frame_count > target_frames:
        # 如果视频过长，均匀采样90帧
        frame_indices = np.linspace(0, original_frame_count - 1, target_frames, dtype=int)
    else:
        # 如果视频过短，使用插值方法补充至90帧
        frame_indices = np.linspace(0, original_frame_count - 1, target_frames)
    
    # 创建视频写入器，设置输出视频的帧率为 30fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (int(cap.get(3)), int(cap.get(4))))
    
    frames = []
    
    # 读取视频帧
    for i in range(original_frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    # 处理帧数据
    processed_frames = []
    for idx in frame_indices:
        if isinstance(idx, int):
            processed_frames.append(frames[idx])  # 直接采样
        else:
            # 如果是浮动索引，进行线性插值
            idx1 = int(np.floor(idx))
            idx2 = min(int(np.ceil(idx)), original_frame_count - 1)

            alpha = idx - idx1
            interpolated_frame = cv2.addWeighted(frames[idx1], 1 - alpha, frames[idx2], alpha, 0)
            processed_frames.append(interpolated_frame)
    
    # 将处理后的帧写入输出视频
    for frame in processed_frames:
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Processed video saved at {output_video_path}")

def process_videos_in_folder(input_folder, output_folder, target_fps=30, target_frames=90):
    # 遍历输入文件夹中的所有目录及文件
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith('.mp4'):  # 只处理 mp4 格式的视频
                # 构建输入视频文件路径
                input_video_path = os.path.join(root, filename)
                
                # 构建输出视频的路径，保持相同的目录结构
                relative_path = os.path.relpath(root, input_folder)
                output_video_folder = os.path.join(output_folder, relative_path)
                os.makedirs(output_video_folder, exist_ok=True)  # 创建输出目录
                
                output_video_path = os.path.join(output_video_folder, filename)  # 输出视频路径

                # 打印正在处理的视频文件名
                print(f"Processing file: {filename}")
                
                # 处理视频
                process_video(input_video_path, output_video_path, target_fps, target_frames)

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Process videos and resize them to a fixed number of frames.")
    
    # 添加输入和输出文件夹路径的命令行参数
    parser.add_argument('--input_folder', type=str, required=True, help="Input folder containing the videos")
    parser.add_argument('--output_folder', type=str, required=True, help="Output folder to save the processed videos")
    
    # 添加目标帧率和目标帧数的命令行参数
    parser.add_argument('--target_fps', type=int, default=30, help="Target FPS for output videos (default: 30)")
    parser.add_argument('--target_frames', type=int, default=90, help="Target number of frames for output videos (default: 300)")
    
    # 解析命令行参数
    args = parser.parse_args()

    # 调用处理视频函数
    process_videos_in_folder(args.input_folder, args.output_folder, args.target_fps, args.target_frames)

if __name__ == "__main__":
    main()