import os
import cv2

def get_video_frame_count(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频的帧数
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 释放视频对象
    cap.release()
    
    return frame_count

def process_videos_in_folder(input_folder):
    # 遍历文件夹中的所有视频文件
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(('.mp4', '.avi', '.mov')):  # 只处理视频文件
                video_path = os.path.join(root, filename)
                
                # 获取视频的帧数
                frame_count = get_video_frame_count(video_path)
                
                # 输出视频的文件名和帧数
                print(f"Video: {filename}, Frame Count: {frame_count}")

if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = '/mnt/pami26/zengyi/dog_action/pose_estimation/data_300/val'  # 替换为你的文件夹路径

    # 调用函数处理文件夹中的所有视频
    process_videos_in_folder(input_folder)