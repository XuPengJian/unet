import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms


# from d2l import torch as d2l

def read_voc_images(voc_dir, is_train=True):
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        img = torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg'))
        img = torchvision.transforms.functional.resize(img, [512, 612], antialias=True)  # 将图片resize到宽612，高512的尺寸
        features.append(img)
        gt = torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass', f'{fname}.png'), mode)
        gt = torchvision.transforms.functional.resize(gt, [512, 612], antialias=False)  # 将图片resize到宽612，高512的尺寸
        labels.append(gt)

    return features, labels


# torchvision.io.read_image 读取之后通道位于第一个维度， 在display 的时候需要将通道数移动到最后一个维度
# train_features, train_labels = read_voc_images(voc_dir, True)

# VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
#                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
#                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
#                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
#                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
#                 [0, 64, 128]]

# VOC_CLASSES = [
#     'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
#     'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128]]

VOC_CLASSES = [
    'background', 'emergency', 'slow', 'fast_1', 'fast_2', 'fast_3']


# # 构建从RGB 到VOC类别索引的映射
def voc_colormap2label():
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 +
                       colormap[2]] = i
    return colormap2label


# 将标签中点的RGB值映射到类别索引
def voc_label_indices(colormap, colormap2label):  # 将colormap 是channel * height * width
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = (colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2]
    #     print(idx.shape)
    return colormap2label[idx]


# y = voc_label_indices(train_labels[0], voc_colormap2label())
# print(y[480:512, 580:612])


# TODO: 图像增广（数据增强）还可以考虑其它的数据增强方法，比如颜色的增强hsv变换

# 随机裁剪数据增强
def voc_rand_crop(feature, label, height, width):
    """random crop 特征和标签"""
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label


def apply_hsv_transform(image, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    """
    对输入的图片进行HSV变换的数据增强
    Args:
    - image: 输入的图片张量，格式为[C, H, W]
    - h_gain: 色调增益
    - s_gain: 饱和度增益
    - v_gain: 明亮度增益
    Returns:
    - transformed_image: 经过HSV变换的图片张量
    """
    # 将图片转换为PIL格式
    pil_image = transforms.ToPILImage()(image)
    # pil_image = Image.fromarray(image.permute(1, 2, 0).numpy())

    # 定义HSV变换
    hsv_transform = transforms.ColorJitter(hue=h_gain, saturation=s_gain, brightness=v_gain)

    # 应用HSV变换
    transformed_pil_image = hsv_transform(pil_image)

    # 将PIL格式的图片转换为张量
    transformed_image = transforms.ToTensor()(transformed_pil_image)

    return transformed_image


# 用户自定义dataset class
# 至少需要实现，init, getitem, len
# 图片分割不好用resize，因为对label进行resize 会有歧义。但可以使用crop
class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [feature for feature in self.filter(features)]  # 经过filter 和 normalize
        self.labels = self.filter(labels)

        self.colormap2label = voc_colormap2label()
        print(f'read {len(self.features)} examples')

    def normalize_image(self, img):
        return self.transform(img.float())

    def filter(self, imgs):  # 由于一些图片的大小比crop_size 的图片还要小
        return [img for img in imgs if (img.shape[1] >= self.crop_size[0] and img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        feature = apply_hsv_transform(feature)
        # feature = feature.float()/255
        # self.normalize_image(feature)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


# 组合为一个函数
def load_data_voc(voc_dir, batch_size, crop_size, num_workers):
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers)

    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size, drop_last=True,
        num_workers=num_workers)
    return train_iter, test_iter


if __name__ == '__main__':
    voc_dir = r'C:\datasets\Segmentation\2017497_1706153604'
    crop_size = (480, 480)
    batch_size = 64
    num_workers = 8
    train_iter, test_iter = load_data_voc(voc_dir, batch_size, crop_size, num_workers)
    for X, Y in train_iter:
        print(X.shape)
        print(Y.shape)
        print(X)
        break
