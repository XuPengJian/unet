import os
import cv2
import argparse
import numpy as np
import ezdxf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cad_file_path', type=str,
                        default=r'',
                        # default=r'D:\裂缝监测\检测报告\新光\新光快速桥梁及隧道不检测平面.dxf',
                        help='上传已有CAD的文件，如果为空的话，默认为新建一个空CAD文件，叫做default')
    # TODO:这里传入的image_path的文件夹里的裂缝需要带上裂缝类型和对应的经纬度坐标，这样才好还原到CAD的坐标上
    parser.add_argument('--image_path', type=str,
                        default=r"D:\gitlab\road_damage_detection\yolov5\predict_result\img_crop_seg",
                        help='检测图片的文件夹路径')
    parser.add_argument('--save_path', type=str,
                        default=r'D:\gitlab\road_damage_detection\yolov5\predict_result\img_crop_cad',
                        help='预测的保存文件夹路径，所有检测结果都会保存在这个路径下')

    # parser.add_argument('--input_shape', type=list, default=[640, 640],
    #                     help='输入网络的分辨率大小[h,w]')  # [1152, 2048]
    return parser.parse_args()


def main(args):
    # 如果CAD路径存在，则打开已存在的DXF文件
    if args.cad_file_path:
        dwg = ezdxf.readfile(args.cad_file_path)
        # 获取模型空间
        msp = dwg.modelspace()
        # 获取保存CAD的名字
        save_name = args.cad_file_path.split('\\')[-1]

    # 如果CAD路径不存在，则保存一个新的CAD路径
    else:
        # 创建一个新的DXF文件
        dwg = ezdxf.new(dxfversion='R2000')
        # 获取模型空间
        msp = dwg.modelspace()
        # 默认保存CAD的名字
        save_name = 'default.dxf'

    # TODO:是否要基于裂缝类型来划分图层(可以根据crop的图片的命名规则来定义)
    # 比如这里可以分4个图层
    layer1 = dwg.layers.new('纵向裂纹')
    layer2 = dwg.layers.new('横向裂纹')
    layer3 = dwg.layers.new('龟裂网裂')
    layer4 = dwg.layers.new('坑槽')
    layer5 = dwg.layers.new('其它裂纹')

    # 设置图层的属性
    # 纵向裂纹图层属性
    layer1.color = 1  # 设置图层的颜色为红色
    layer1.linetype = 'Continuous'  # 设置图层的线型为实线
    # 横向裂纹图层属性
    layer2.color = 1  # 设置图层的颜色为红色
    layer2.linetype = 'Continuous'  # 设置图层的线型为实线
    # 龟裂网裂图层属性
    layer3.color = 6  # 设置图层的颜色为洋红色
    layer3.linetype = 'Continuous'  # 设置图层的线型为实线
    # 坑槽图层属性
    layer4.color = 2  # 设置图层的颜色为黄色
    layer4.linetype = 'Continuous'  # 设置图层的线型为实线
    # 其它图层属性
    layer5.color = 7  # 设置图层的颜色为白色
    layer5.linetype = 'Continuous'  # 设置图层的线型为实线
    # layer.linetype = 'DASHED'  # 设置图层的线型为虚线

    # 遍历路径下的每一张图片
    image_files = os.listdir(args.image_path)
    for i, file_name in enumerate(image_files):
        if file_name.endswith(".png") or file_name.endswith(".jpg"):

            # 基于文件名所带的类型名字来选择绘制的结果放在哪个图层
            cls = file_name.split('.')[0].split('_cls_')[-1]
            if int(cls) == 0:
                layer = layer1
            elif int(cls) == 1:
                layer = layer2
            elif int(cls) == 2:
                layer = layer3
            elif int(cls) == 3:
                layer = layer4
            else:
                layer = layer5

            image_path = os.path.join(args.image_path, file_name)
            # 处理每个图像文件
            image = cv2.imread(image_path, 0)  # 以灰度图像方式读取

            # 使用高斯模糊(对边缘做一定的模糊处理, 让边缘不要过于锯齿状)
            image = cv2.GaussianBlur(image, (5, 5), 0)  # 使用5x5的高斯滤波器进行平滑

            # 使用Canny边缘检测算法
            edges = cv2.Canny(image, 100, 200)

            # 测试时查看图片的质量
            # cv2.imshow(f"{file_name}", image)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # 生成边缘角点坐标
            # 这些是相关的设置

            # mode
            # cv2.RETR_EXTERNAL只检测外轮廓
            # cv2.RETR_LIST检测的轮廓不建立等级关系
            # cv2.RETR_CCOMP建立两个等级的轮廓
            # cv2.RETR_TREE建立一个等级树结构的轮廓

            # method
            # cv2.CHAIN_APPROX_SIMPLE = 2
            # cv2.CHAIN_APPROX_TC89_L1 = 3
            # cv2.CHAIN_APPROX_TC89_KCOS = 4

            contours, _ = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            # print(contours)

            # 绘制轮廓
            for contour in contours:
                # 将每个轮廓的点添加到多段线(平移一下,方便看~)
                # 每个先移动1000,方便前期调试查看
                # TODO:结合裂缝的实际尺寸进行缩放和经纬度和坐标位置转换
                distance = i * 1000
                points = [(point[0][0] + distance, -point[0][1]) for point in contour]
                # points.append(points[0])  # 将最后一个点重复添加到多段线

                # polyline = msp.add_lwpolyline(points)
                # 创建一个新的多段线实体
                polyline = msp.add_lwpolyline(points)

                # 将多段线实体的图层属性设置为所需的图层
                polyline.dxf.layer = layer.dxf.name

    # 在绘制完所有线之后,保存DXF文件
    dwg.saveas(os.path.join(args.save_path, save_name))


if __name__ == "__main__":
    args = parse_args()
    main(args)
