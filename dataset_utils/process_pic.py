import os
import cv2

# 图片路径
image_path = r'C:\datasets\RoadDamageDatasets\CrackForest\label'

# 遍历路径下的所有文件
for filename in os.listdir(image_path):
    # 判断文件是否为图片
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # 读取图片
        image = cv2.imread(os.path.join(image_path, filename))

        # 将图片的值乘以255
        image = image * 255

        # 保存图片
        cv2.imwrite(os.path.join(image_path, filename), image)