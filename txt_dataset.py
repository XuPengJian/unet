import math
import warnings

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data.dataset import Dataset
import random

'''
随机水平翻转
'''


def random_horizontal_flip(image, gt, threshold=0.5):
    v = random.random()
    if v < threshold:
        image = cv2.flip(image, 1)
        gt = cv2.flip(gt, 1)
    return image, gt


'''
随机竖直翻转
'''


def random_vertical_flip(image, gt, threshold=0.5):
    v = random.random()
    if v < threshold:
        image = cv2.flip(image, 0)
        gt = cv2.flip(gt, 0)
    return image, gt


'''
随机HSV增强
'''


# def random_hsv(image, hgain=0.1, sgain=0.7, vgain=0.3):
def random_hsv(image, hgain=0.4, sgain=1.0, vgain=0.6):
    '''
    img: 输入的图片，需要为numpy.array格式
    hgain: H的随机扰动比例
    sgain: S的随机扰动比例
    vgain: V的随机扰动比例
    -------------
    return: numpy.array
    '''
    # 将图片从RGB转换为HSV，好处是通过变换S和V可以改变亮度和饱和度，做到数据增强
    # 随机取三个[-1, 1)的值，乘以输入的[hgain, sgain, vgain]再加1，这里获取的是三个1左右的比值
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    # 将图像转到HSV上
    # H表示色调，取值范围是[0,180]
    # S表示饱和度，取值范围是[0,255]
    # V表示亮度，取值范围是[0,255]
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    dtype = image.dtype
    # 应用变换
    # x=[0, 1 ... 255]
    x = np.arange(0, 256, dtype=r.dtype)
    # 对H值添加扰动，这个扰动一般较小，H值最大为180，所以要对180取余
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    # 对S值添加扰动
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    # 对V值添加扰动
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    # cv2.LUT(src, lut, dst=None) LUT是look-up table(查找表)的意思
    # src：输入的数据array，类型为8位整型（np.uin8)
    # lut：查找表
    # 这里cv2.LUT(hue, lut_hue)的作用就是将hue原来的值作为索引去lut_hue中找到对应的新值，然后赋给hue
    # 比如在hue中有个值是100，则取lut_hue[100]作为hue当前位置的新值
    # cv2.merge:合并通道，不用指定维度，但是这个操作比较耗时，所以改用np.stack
    # image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    image_data = np.stack((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)), axis=2)
    # HSV --> RGB
    image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
    return image_data


'''
randomcrop数据增强
'''


def random_crop(rgb, gt, crop_size, fixed=False):
    # if scale[0] > scale[1]:
    #     warnings.warn("Scale and ratio should be of kind (min, max)")
    # 获取目标宽高
    target_w, target_h = crop_size
    # 获取图片宽高并求面积
    h, w = rgb.shape[:2]
    area = h * w
    # 计算宽高比
    ratio = w / h

    '''
    做一个图片大小的判断
    原图w或h小于目标裁剪wh大小时按原图等比例放大
    原图面积大于目标面积的4倍时等比缩小至原图面积约1/2
    '''
    # if h < target_h or w < target_w:
    #     mag_ratio = math.ceil(max(target_h / h, target_w / w))
    #     w = w * mag_ratio
    #     h = h * mag_ratio
    #     rgb = cv2.resize(rgb, (w, h))
    #     gt = cv2.resize(gt, (w, h))
    # while area > target_h * target_w * 4:
    #     w = int(w * 0.7)
    #     h = int(h * 0.7)
    #     rgb = cv2.resize(rgb, (w, h))
    #     gt = cv2.resize(gt, (w, h))
    #     area = h * w

    # # 通过比例计算裁剪目标图片的面积大小
    # scale_a = random.uniform(scale[0], scale[1])
    # target_area = area * scale_a
    #
    # # 通过计算面积与宽高比计算宽高
    # target_w = int(round(math.sqrt(target_area * ratio)))
    # target_h = int(round(math.sqrt(target_area / ratio)))
    # # 设置尺寸为32的倍数
    # target_w = 32 * (target_w // 32)
    # target_h = 32 * (target_h // 32)

    # 确保裁剪区域不超出图像范围
    max_w = w - target_w
    max_h = h - target_h

    # 验证机固定裁剪区域在中央
    if fixed:
        x = 0
        y = 112
    else:
        # 随机生成裁剪起始点坐标
        x = random.randint(0, max_w)
        y = random.randint(0, max_h)

    # 执行裁剪操作
    cropped_rgb = rgb[y:y + target_h, x:x + target_w]
    cropped_gt = gt[y:y + target_h, x:x + target_w]

    return cropped_rgb, cropped_gt


"""
随机高斯模糊
"""


def random_gaussian(image, threshold=0.5):
    v = random.random()
    if v < threshold:
        ksize = (5, 5)  # 模糊核的大小
        sigma = np.random.uniform(0.1, 5.0)  # 随机生成标准差
        image = cv2.GaussianBlur(image, ksize, sigma)
    return image


"""
随机高斯噪声
"""


def random_gaussian_noise(image, threshold=0.5):
    v = random.random()
    if v < threshold:
        # 添加高斯噪声
        mean = 0  # 噪声均值
        var = np.random.uniform(0.001, 0.01)  # 随机生成方差
        sigma = var ** 0.5  # 计算标准差
        noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)
    return image


'''
语义分割数据集构建
'''


class MaskDataset(Dataset):
    def __init__(self, data_path, input_shape=None, mode='train', transforms=None):
        '''
        :param data_path: 一个txt，每一行对应一张图(origin_image_path 空格 depth_image_path)
        :param input_shape: 输入分辨率大小
        '''
        super(MaskDataset, self).__init__()
        with open(data_path, 'r') as f:
            self.data = f.readlines()
        self.input_shape = input_shape
        self.length = len(self.data)
        self.epoch_now = -1
        self.mode = mode
        # 用于存储目标裁剪宽高的列表
        self.crop_size = input_shape
        self.transforms = transforms
        # # 设置一个随机生成的size，作为裁剪的大小
        # target_w = random.uniform(32 * 6, 32 * 10)
        # target_h = random.uniform(32 * 6, 32 * 10)
        # # 生成宽高（32的倍数）
        # self.crop_size.append(int(32 * (target_w // 32)))
        # self.crop_size.append(int(32 * (target_h // 32)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = self.data[index].split('\n')[0].split(',')[0]
        gt_path = self.data[index].split('\n')[0].split(',')[1]

        img = Image.open(img_path).convert('RGB')
        manual = Image.open(gt_path).convert('L')
        mask = np.array(manual) / 255
        img = np.array(img)

        # 训练集采用数据增强
        if self.mode == 'train':
            # 水平随机翻转
            # img_rgb, img_mask = random_horizontal_flip(img_rgb, img_mask)
            # 垂直随机翻转
            # img_rgb, img_mask = random_vertical_flip(img_rgb, img_mask)
            img = random_hsv(img)
            # 使用random_crop的操作
            # img_rgb, img_depth = random_crop(img_rgb, img_depth, self.crop_size)
            # 使用随机高斯模糊
            img = random_gaussian(img)
            # 使用随机高斯噪声
            img = random_gaussian_noise(img)

            # 使用resize操作
            img = cv2.resize(img, self.input_shape)
            mask = cv2.resize(mask, self.input_shape)
        # 验证集整张图预测，但要防止显存爆炸
        if self.mode == 'val':
            # 使用random_crop的操作
            # img_rgb, img_depth = random_crop(img_rgb, img_depth, self.crop_size, fixed=True)
            # 使用resize的操作
            img = cv2.resize(img, self.input_shape)
            mask = cv2.resize(mask, self.input_shape)
            # img_rgb = cv2.resize(img_rgb, (512, 288))

        if self.mode == 'test':
            # 只对输入的原图进行resize，gt不进行resize，用于更准确的评估依据
            img = cv2.resize(img, self.input_shape)
            # img_depth = cv2.resize(img_depth, self.input_shape)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)
        img = Image.fromarray(img)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
# DataLoader中collate_fn使用
# def depth_dataset_collate(batch):
#     images = []
#     gts = []
#     for img, gt in batch:
#         images.append(img)
#         gts.append(gt)
#     images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
#     gts = torch.from_numpy(np.array(gts)).type(torch.FloatTensor)
#     return images, gts
