import os
import glob
import numpy as np
import cv2
from datetime import datetime
import pickle
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
auto_seg = True

# 从指定目录中的视频文件中提取帧，并将这些帧处理成适合深度学习模型输入的数据格式

def generate_data_json(video_dir: str, json_path: str, max_seq_len, frame_rate: int, video_res, abandon_len):
    classifications = [directory for directory in os.listdir(video_dir) if os.path.isdir(video_dir+directory)]
    labels = []
    videos = []
    label_map = {}
    # no truncate or padding if max_seq_len is 0
    if max_seq_len != 0:
        cut_len = max_seq_len
    else:
        cut_len = np.inf

    for index, classification in enumerate(classifications):
        path = video_dir + classification
        label_map[index] = classification
        for video in glob.glob(path+"/*.mp4"):
            print(video)

            # extract frames from video into np array
            video_arr = []
            vidcap = cv2.VideoCapture(video)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            # 需要指定一个帧率
            stride = round(fps/frame_rate)
            success, image = vidcap.read()
            frame_cnt = 1
            while success: # 可以读取下一帧
                if fps > 240:
                    vidcap.set(1, (stride*1000))
                else:
                    vidcap.set(1, frame_cnt)
                # 如果视频帧率 fps 超过 240，每隔 stride * 1000 毫秒读取一帧。
                # 否则，每隔 stride 帧读取一帧。
                # 增加帧计数器 frame_cnt，跳过 stride 帧。

                frame_cnt += stride # 要读的是第几帧
                image = cv2.resize(image, video_res, interpolation=cv2.INTER_AREA) # 调整分辨率
                # cv2.imshow("image", image)
                # cv2.waitKey()
                # image = (image - 255 * 0.5) / (255 * 0.5)
                mean = np.mean(image)
                stdDev = np.std(image)
                image = (image - mean) / stdDev  # replicate pytorch normalize

                video_arr.append(image)
                # 序列中加入图片
                # truncate video if video len >= max len
                if len(video_arr) >= cut_len:
                    if auto_seg:  # segment video automatically
                        videos.append(video_arr)
                        labels.append(index)
                        video_arr = []
                    else:
                        break
                # load next frame
                success, image = vidcap.read()

            # padding if vid len < max len
            original_len = len(video_arr)
            while len(video_arr) < max_seq_len:
                video_arr.append(np.zeros(video_res+(3,)))
            print(f"fps: {fps}, stride: {stride}, len:{frame_cnt}")
            if original_len >= abandon_len:
                videos.append(video_arr)
                labels.append(index)
            else:
                print(f"abandoned len of: {original_len}")

    video_info = {}
    if max_seq_len != 0:
        videos = np.array(videos)
        # print("videos:", videos.shape)
        videos = np.transpose(videos, (0, 4, 1, 2, 3))
        video_info = {"input_channels": videos.shape[1],  "seq_len": videos.shape[2], "height": videos.shape[3], "width": videos.shape[4]}
    else:
        # for seq of varying lengths because of no padding
        videos = np.array(videos, dtype=object)

    print(f"video shape: {videos.shape}")
    # print(f"labels: {labels}")
    print(f"label_map: {label_map}")

    video_info["num_classes"] = len(classifications)
    data = {"videos": videos, "labels": labels, "label_map": label_map, "video_info": video_info}

    label_cnt = {i: labels.count(i) for i in labels}
    label_name_cnt = dict((label_map.get(k, k), v) for (k, v) in label_cnt.items())  # count number of every class
    print(label_name_cnt)

    # write to pkl
    with open(json_path, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    start = datetime.now()
    print(f"device: {device}")

    video_dir = "data/Doge_dataset/"
    save_dir = "data/Doge_dataset/data128.pkl"
    generate_data_json(video_dir, save_dir, max_seq_len=15, frame_rate=25, video_res=(128, 128), abandon_len=6)

    end = datetime.now()
    print(f"執行時間：{end - start}")
