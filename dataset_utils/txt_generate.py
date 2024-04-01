import os

# rgb_path = r"C:\datasets\RoadDamageDatasets\CrackForest\image"
# mask_path = r"C:\datasets\RoadDamageDatasets\CrackForest\label"

rgb_path = r"C:\datasets\RoadDamageDatasets\CrackMap\images"
mask_path = r"C:\datasets\RoadDamageDatasets\CrackMap\masks"

save_path = r'D:\gitlab\unet\datasets'

rgb_list = []
mask_list = []

# 获取数据集名称创建对应txt文件
txt_name = rgb_path.split('\\')[-2]
# 创建存放txt数据的文件夹
if not os.path.exists(save_path):
    os.makedirs(save_path)
txt_file = os.path.join(save_path, txt_name + '.txt')

for root, dirs, files in os.walk(mask_path):
    if files != '' and files != []:
        for i, file in enumerate(files):
            files[i] = os.path.join(root, file)
        mask_list.extend(files)
for root, dirs, files in os.walk(rgb_path):
    if files != '' and files != []:
        for i, file in enumerate(files):
            files[i] = os.path.join(root, file)
        rgb_list.extend(files)

# 新建空白txt
with open(txt_file, "w") as file:
    file.write('')

# 通过for循环续写
for rgb, depth in zip(rgb_list, mask_list):
    file_name = rgb + ',' + depth
    with open(txt_file, "a") as file:
        file.write(file_name + '\n')
        print(file_name)
    file.close()

