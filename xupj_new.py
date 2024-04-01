import numpy as np
import cv2

image_path = r'C:\datasets\Segmentation\2017497_1706153604\SegmentationClass\UGSMT500GC_1_20231205155428487.png'

# 设置新的图像大小
new_width = 600
new_height = 500

# 读取图片
image = cv2.imread(image_path)
# 调整图像大小
image = cv2.resize(image, (new_width, new_height))

mask = np.all(image == [128, 0, 0], axis=2)

# 将掩模转换为8位无符号整型
mask = mask.astype(np.uint8) * 255

# 显示原始图片和掩模
cv2.imshow('Original Image', image)
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
