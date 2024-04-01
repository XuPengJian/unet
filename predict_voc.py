import os
import time

import cv2
import argparse
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


'''
推理超参设置
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=int, default=5,
                        help='exclude background')
    parser.add_argument('--weights_path', type=str,
                        default=r"./save_weights/voc_best_model.pth",
                        help='模型权重路径')
    parser.add_argument('--image_folder', type=str,
                        default=r"C:\datasets\Segmentation\2017497_1706153604\JPEGImages",
                        # default=r"C:\datasets\Segmentation\images",
                        help='检测图片的文件夹路径')
    # parser.add_argument('--save_folder', type=str,
    #                     default=r'D:\gitlab\road_damage_detection\yolov5\predict_result\road_seg',
    #                     help='预测的保存文件夹路径，所有检测结果都会保存在这个路径下')
    # parser.add_argument('--input_shape', type=list, default=[640, 640],
    #                     help='输入网络的分辨率大小[h,w]')  # [1152, 2048]
    return parser.parse_args()


def main(args):
    assert os.path.exists(args.weights_path), f"weights {args.weights_path} not found."
    assert os.path.exists(args.image_folder), f"image {args.image_folder} not found."
    # save_folder = args.save_folder
    save_folder = args.image_folder.replace('images', 'outputs') + '/unet_output'
    os.makedirs(save_folder, exist_ok=True)

    # mean = (0.709, 0.381, 0.224)
    # std = (0.127, 0.079, 0.043)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = UNet(in_channels=3, num_classes=args.classes + 1, base_c=32)

    # load weights
    model.load_state_dict(torch.load(args.weights_path, map_location='cpu')['model'])
    model.to(device)

    # load roi mask
    # roi_img = Image.open(roi_mask_path).convert('L')
    # roi_img = np.array(roi_img)

    # 逐个遍历
    for filename in tqdm(os.listdir(args.image_folder)):
        # 判断是否为图片文件
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            # 拼接文件路径
            img_path = os.path.join(args.image_folder, filename)
            # load image
            img = torchvision.io.read_image(img_path)
            c, org_h, org_w = img.shape
            img = torchvision.transforms.functional.resize(img, [512, 612], antialias=True)  # 将图片resize到宽612，高512的尺寸

            data_transform = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
            # from pil image to tensor and normalize
            # original_img = Image.open(img_path).convert('RGB')
            # original_size = original_img.size
            # original_img = original_img.resize((512, 512))
            # from pil image to tensor and normalize
            # data_transform = transforms.Compose([transforms.ToTensor(),
            #                                      transforms.Normalize(mean=mean, std=std)])

            # img = data_transform(img.float())
            # 需要先将图片处理为0-1之间
            img = img.float() / 255
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            model.eval()  # 进入验证模式
            with torch.no_grad():
                # init model
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)

                # t_start = time_synchronized()
                output = model(img.to(device))
                # t_end = time_synchronized()
                # print("inference time: {}".format(t_end - t_start))

                prediction = output['out'].argmax(1).squeeze(0)
                prediction = prediction.to("cpu").numpy().astype(np.uint8)

                # 缩放回原始尺寸，写在同个循环里
                prediction = cv2.resize(prediction, (org_w, org_h), interpolation=cv2.INTER_NEAREST)
                # 这个地方通过索引获取道路分割结果，先h再w
                # pixel_h = 2000
                # pixel_w = 500
                #
                # cls = prediction[pixel_h, pixel_w]
                # print(cls)

                # 将前景对应的像素值改成255(白色)
                prediction = prediction * 51
                # prediction[prediction == 1] = 255
                # prediction[prediction == 2] = 204
                # prediction[prediction == 3] = 153
                # prediction[prediction == 4] = 102
                # prediction[prediction == 5] = 51
                # 将不敢兴趣的区域像素设置成0(黑色)
                # prediction[roi_img == 0] = 0
                mask = Image.fromarray(prediction)
                # mask = mask.resize(original_size)
                mask.save(os.path.join(save_folder, filename.split('.')[0] + '.png'))
            # break


if __name__ == "__main__":
    args = parse_args()
    main(args)
