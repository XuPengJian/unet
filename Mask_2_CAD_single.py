import os
import cv2
import numpy as np
import ezdxf

folder_path = r"D:\gitlab\road_damage_detection\yolov5\predict_result\img_crop_seg"
output_path = r'D:\gitlab\road_damage_detection\yolov5\predict_result\img_crop_cad'
cad_file_path = r'D:\裂缝监测\test.dxf'


# 遍历路径下的每一张图片
image_files = os.listdir(folder_path)
for file_name in image_files:
    if file_name.endswith(".png") or file_name.endswith(".jpg"):
        image_path = os.path.join(folder_path, file_name)
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

        # # 打开已存在的DXF文件
        # dwg = ezdxf.readfile(cad_file_path)
        # # 获取模型空间
        # msp = dwg.modelspace()

        # 创建一个新的DXF文件
        dwg = ezdxf.new(dxfversion='R2000')
        # 获取模型空间
        msp = dwg.modelspace()

        # 创建一个新的图层
        # TODO:是否要基于裂缝类型来划分图层
        layer = dwg.layers.new('CrackContourLayer')
        # 设置图层的属性
        layer.color = 2  # 设置图层的颜色为黄色
        layer.linetype = 'Continuous'  # 设置图层的线型为虚线
        # layer.linetype = 'DASHED'  # 设置图层的线型为虚线

        # TODO:结合裂缝的实际尺寸进行缩放和经纬度和坐标位置转换

        # 绘制轮廓
        for contour in contours:
            # 将每个轮廓的点添加到多段线
            points = [(point[0][0], -point[0][1]) for point in contour]
            # points.append(points[0])  # 将最后一个点重复添加到多段线

            # polyline = msp.add_lwpolyline(points)
            # 创建一个新的多段线实体
            polyline = msp.add_lwpolyline(points)

            # 将多段线实体的图层属性设置为所需的图层
            polyline.dxf.layer = layer.dxf.name

        # 保存DXF文件
        dwg.saveas(os.path.join(output_path, file_name.split('.')[0] + ".dxf"))
