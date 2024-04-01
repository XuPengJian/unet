import os
import cv2

filepath = r'C:\datasets\RoadDamageDatasets\CrackForest\label\001.png'
filepath = r'C:\datasets\RoadDamageDatasets\CrackMap\masks\GOPR0104 (1).png'
filepath = r'C:\datasets\RoadDamageDatasets\CrackMap\images\GOPR0104 (1).png'
mask = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
print(mask.min())
print(mask.max())

import cv2
import numpy as np

# 读取图片
image = cv2.imread(filepath)

# 添加高斯模糊
ksize = (5, 5)  # 模糊核的大小
sigma = np.random.uniform(0.1, 5.0)  # 随机生成标准差
blurred = cv2.GaussianBlur(image, ksize, sigma)

# 添加高斯噪声
mean = 0  # 噪声均值
var = np.random.uniform(0.001, 0.01)  # 随机生成方差
sigma = var ** 0.5  # 计算标准差
noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)

# 显示原始图片、模糊后的图片和添加噪声后的图片
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred)
cv2.imshow('Noisy Image', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()