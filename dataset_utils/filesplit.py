import os
import random

import numpy as np

# 固定随机种子
random.seed(60)

class SplitFiles():
    """按行分割文件"""
    # param: data_file——需合并的几个txt所在文件夹的路径，通过parser作为参数输入
    # param: 按照比例分割训练集与测试集，默认训练集：测试集=7:3
    def __init__(self, data_file, alpha=0.9):
        """初始化要分割的源文件名和分割后的文件行数"""
        self.data_file = data_file
        self.alpha = alpha

    def merge_file(self):
        # file = open(self.file_name, 'w', encoding='utf8')
        filenames = os.listdir(self.data_file)
        txt_list = []
        train_list = []
        val_list = []
        for filename in filenames:
            filepath = os.path.join(self.data_file, filename)
            # 遍历单个文件，读取行数
            temp_tables = []
            f = open(filepath, "r", encoding='utf-8')
            line = f.readlines()
            f.close()
            temp_tables.extend(line)
            txt_list.append(temp_tables)
        # 按比例分为训练集与测试集写入对应列表
        for i in range(len(txt_list)):
            # 对列表中每部分通过shuffle打乱
            random.shuffle(txt_list[i])
            # 获取索引对应部分列表的长度
            num = len(txt_list[i])
            # 通过输入参数作为比例计算该部分下训练集的数量
            offset = int(num * self.alpha)
            # 将列表分为两部分分别写入对应的列表中
            train = txt_list[i][:offset]
            val = txt_list[i][offset:]
            train_list.extend(train)
            val_list.extend(val)
            print(len(train_list))
        # 调用函数将列表写入对应txt文件中
        self.write_file('train', train_list)
        self.write_file('val', val_list)

    def get_part_file_name(self, part_name):
        # 获取分割后的文件名称：在源文件相同目录下建立临时文件夹temp_part_file，然后将分割后的文件放到该路径下
        temp_path = os.path.join(os.getcwd())  # 获取文件的路径（不含文件名）
        file_folder = os.path.join(temp_path, "../temp_part_file")
        if not os.path.exists(file_folder):  # 如果临时目录不存在则创建
            os.makedirs(file_folder)
        part_file_name = os.path.join(file_folder, str(part_name) + ".txt")
        return part_file_name

    def write_file(self, part_num, part_list):
        # 传入参数为名称及其对应的列表
        # 获取对应文件名
        part_file_name = self.get_part_file_name(part_num)
        # 打开对应txt文件并写入
        with open(part_file_name, "w") as fp:
            fp.writelines(part_list)
            fp.close()


if __name__ == "__main__":
    splitfile = SplitFiles(r"D:\gitlab\unet\datasets")
    splitfile.merge_file()

