import os
import random

# 设置路径
folder_path = r'C:\datasets\Segmentation\2017497_1706153604\SegmentationClass'  # 替换为你的文件夹路径
train_ratio = 0.9  # 训练集比例
val_ratio = 0.1  # 验证集比例

# 获取文件夹下所有图片的文件名
file_names = os.listdir(folder_path)

# 去掉文件名的后缀名
file_names = [os.path.splitext(file_name)[0] for file_name in file_names]

# 随机打乱文件名顺序
random.shuffle(file_names)

# 计算划分的索引
train_index = int(len(file_names) * train_ratio)

# 划分训练集和验证集
train_file_names = file_names[:train_index]
val_file_names = file_names[train_index:]

# 将文件名写入train.txt
with open('train.txt', 'w') as f:
    for file_name in train_file_names:
        f.write(file_name + '\n')

# 将文件名写入val.txt
with open('val.txt', 'w') as f:
    for file_name in val_file_names:
        f.write(file_name + '\n')