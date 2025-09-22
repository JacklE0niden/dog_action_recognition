import torch
import os
import pickle
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np

import cv2
import glob
import re

data_tf = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize((0.5, ), (0.5, ))
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)


class ImageDataset(Dataset):
    def __init__(self, data_file):
        f = open(data_file, "rb") 
        data = pickle.load(f)
        self.videos = data["videos"]
        self.labels = data["labels"]
        self.label_map = data["label_map"]
        self.input_shape = data["video_info"]
        self.transform = data_tf

        print(data["video_info"])
        print("label map", self.label_map)
        label_cnt = {i: self.labels.count(i) for i in self.labels}
        label_name_cnt = dict((data["label_map"].get(k, k), v) for (k, v) in label_cnt.items())  # count number of every class
        print("label count", label_name_cnt)

        data_size = len(self.labels)
        self.balanced_weight = [data_size/i for i in label_cnt.values()]  # calculate balanced weight
        print("self.balanced_weight:", self.balanced_weight)
    # The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # videos = self.transform(self.videos[index])
        videos = self.videos[index]
        targets = torch.tensor(self.labels[index])
        sample = {"videos": videos, "labels": targets}
        return sample
    



# 定义视频处理的Dataset类
class VideoDataset(Dataset):
    def __init__(self, pkl_path):
        """
        初始化数据集，读取pkl文件中的数据
        :param pkl_path: 存放pkl文件的路径
        """
        self.pkl_path = pkl_path
        self.videos = []
        self.labels = []
        self.label_map = {}
        self.video_info = {}
        self.balanced_weight = []  # 用于存储每个类别的权重
        self._load_data()

    def _load_data(self):
        # 从pkl文件加载数据
        with open(self.pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        self.videos = data["videos"]  # 从pkl文件加载视频数据
        self.labels = data["labels"]  # 从pkl文件加载标签数据
        self.label_map = data["label_map"]  # 从pkl文件加载标签映射
        self.video_info = data["video_info"]  # 获取视频信息（如帧数、分辨率等）

        # 计算每个类别的样本数
        label_cnt = {label: self.labels.count(label) for label in set(self.labels)}
        total_samples = len(self.labels)

        # 计算每个类别的权重：每个类别样本数的倒数
        self.balanced_weight = [total_samples / label_cnt[label] for label in range(9)]  # 9 类别

        print("self.balanced_weight:", self.balanced_weight)  # 打印类别平衡权重


    def __len__(self):
        """返回数据集的大小"""
        return len(self.labels)

    def __getitem__(self, index):
        """
        获取某个索引处的视频数据和标签
        :param index: 索引
        :return: 视频帧和对应的标签以及样本的权重
        """
        # print("self.labels:", self.labels)
        # print("self.label_map:", self.label_map)
        video = self.videos[index]
        label = self.labels[index]
        # 根据标签值直接获取标签索引
        
        # label_id = list(self.label_map.keys())[list(self.label_map.values()).index(label)]
        label_id = label  # label是数字，直接赋值给label_id
        # print("video.shape:", video.shape)
        # weight = self.balanced_weight[index]  # 获取样本对应的权重


        print("label_id:", label_id)
        weight = self.balanced_weight[label_id]  # 获取样本对应的权重

        return {"video": torch.tensor(video, dtype=torch.float32), 
                "label": torch.tensor(label_id, dtype=torch.long), 
                "weight": torch.tensor(weight, dtype=torch.float32)}