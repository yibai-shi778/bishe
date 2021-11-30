import random
import glob
import os
import csv
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def process_image(image, target_shape):
    """
    图像缩放与归一化
    数据是拷贝的，因此不会改变文件中的原图大小
    """
    image = Image.open(image)

    transform = transforms.Compose([
        transforms.Resize(target_shape),
        transforms.ToTensor()
    ])

    return transform(image).numpy()


def get_corresponding_labels(sample_path):
    """
    根据样本路径查找相应标签，返回的应当是一个整数值而非字符,
    每次都要开关文件是否影响了运行速度？
    """
    cur_dir = sys.path[1]
    sample_id = sample_path.split('\\')[-1]
    train_or_test = sample_path.split('\\')[-3]

    with open(os.path.join(cur_dir, 'labels', 'Re' + train_or_test + 'Labels.csv'), 'r') as f:
        label_list = np.array(list(csv.reader(f)))
        id_list = label_list[:, 0].tolist()
        engagement_list = label_list[:, 1]

    return int(engagement_list[id_list.index(sample_id + '.avi')])


def seq_data_iter_random(batch_size, dataset, datatype, target_shape):
    """使用随机抽样生成一个小批量子序列"""
    round_count = 0
    sample_list = glob.glob(os.path.join(dataset, '*\\*'))
    random.shuffle(sample_list)
    num_batches = int(np.floor(len(sample_list) / batch_size))
    for _ in range(num_batches):
        X, y = [], []
        for i in range(batch_size):
            # 按照我的设想，该sample文件夹中应当存有原视频文件、采样的帧序列和帧序列中提取出的空间特征
            # 首先随机选取一文件夹，具体择出谁作X依据datatype确定
            sample = sample_list[i + round_count * batch_size]

            # CNN_input是图序列，LSTM_input是特征向量序列
            if datatype is 'CNN_input':
                frames = sorted(glob.glob(os.path.join(sample, '*.jpg')))
                sequence = [process_image(frame, target_shape) for frame in frames]
            else:
                sequence = np.load(glob.glob(os.path.join(sample, '*.npy')))

            X.append(sequence)
            # 按理说每一step上都对应着相同的y，是否需要将其扩展为30维的全1或全0向量？
            y.append(get_corresponding_labels(sample))

        X = np.array(X).transpose(0, 2, 1, 3, 4)
        X = torch.tensor(X)
        y = torch.tensor(y)
        round_count += 1

        yield X, y


# 李沐的d2l中的迭代器生成代码，看不懂！
# class SeqDataLoador:
#     def __init__(self, batch_size, dataset, datatype, target_shape):
#         self.data_iter_fn = seq_data_iter_random()
#         self.batch_size, self.dataset, self.datatype, self.target_shape = \
#             batch_size, dataset, datatype, target_shape
#
#     def __iter__(self):
#         return self.data_iter_fn(self.batch_size, self.dataset, self.datatype, self.target_shape)


if __name__ == '__main__':

    batch_size = 10
    test_path = 'E:\\yan0\\bishe_test\\Test'
    datatype = 'CNN_input'
    target_shape = [224, 224]

    for X, y in seq_data_iter_random(batch_size, test_path, datatype, target_shape):
        print(X.shape, y.shape)
        break
