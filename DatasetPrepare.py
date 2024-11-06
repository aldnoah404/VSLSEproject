# 1.数据准备——对标签进行处理：
# 通过中值滤波去除 图像中的椒盐噪声，
# 然后根据图像灰度直方图确定分割阈值，分割出仅包含激光条的二值图像。
# 接着，使用形态学操作对分割的二值图像进行处理，去除激光条纹上的 孔洞和毛刺，完整激光条纹掩膜质量。
# 将改善后的二值图像作为数据集的标注图像。 而样本中对应的有噪声图像则作为数据集的输入。
# 2.创建后续需要使用的数据集

import cv2 as cv
import os
import numpy as np
import shutil
from ToSequence import rm_dir

def dataset_prepare(proto_input_path,proto_target_path,name,threshold):
    # 拟输入： 
    # proto_input_path = r'./CroppedDataset/B/imgs'
    # proto_target_path = r'./CroppedDataset/B/targets'
    # name = 'B'

    # 复制input到新文件夹
    proto_input_path = proto_input_path
    new_input_path = f'preDataset/{name}/inputs'
    print(f'开始将文件夹 {proto_input_path} 中的内容复制到文件夹 {new_input_path} 中')
    rm_dir(new_input_path)
    for item in os.listdir(proto_input_path):
        s = os.path.join(proto_input_path, item)
        d = os.path.join(new_input_path, item)
        shutil.copytree(s, d)
    print("复制完成")

    # 对target进行处理并保存
    proto_target_path = proto_target_path # r'./CroppedDataset/B/targets'
    proto_target_path_list = os.listdir(proto_target_path) # [1,2,3]
    new_target_path = f'preDataset/{name}/targets' # r'Dataset/{name}/targets'
    print(f"开始处理 {proto_target_path} 中的图片并保存到 {new_target_path}")
    rm_dir(new_target_path)

    for path in proto_target_path_list: 
        image_sequence_path = os.path.join(proto_target_path, path)# r'./CroppedDataset/B/targets/1'
        image_paths = os.listdir(image_sequence_path) # 1.jpg
        save_sequence = os.path.join(new_target_path, path)
        rm_dir(save_sequence)
        for img in image_paths:
            img_path = os.path.join(image_sequence_path, img)
            image = cv.imread(img_path, 0)
            # 中值滤波
            blur = cv.medianBlur(image, 5)
            threshold = threshold
            # 阈值分割,这里阈值用的是固定值200
            _, binary_image = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY)

            # 形态学操作去除孔洞和毛刺
            kernel = np.ones((7, 5), np.uint8)
            morphological_image = cv.morphologyEx(binary_image, cv.MORPH_CLOSE, kernel)  # 闭运算

            # 图像保存

            save_path = os.path.join(new_target_path, path, img) # r'Dataset/{name}/targets/1'
            cv.imwrite(save_path, morphological_image)
    print("处理完成")


if __name__ == "__main__":
    # 阈值200
    dataset_prepare(proto_input_path=r'CroppedDataset/B/imgs',
                    proto_target_path=r'CroppedDataset/B/targets',
                    name="B", threshold=200)
    dataset_prepare(proto_input_path=r'CroppedDataset/L/imgs',
                    proto_target_path=r'CroppedDataset/L/targets',
                    name="L", threshold=200)

    # 阈值100
    dataset_prepare(proto_input_path=r'CroppedDataset/B_C/imgs',
                    proto_target_path=r'CroppedDataset/B_C/targets',
                    name="B_C", threshold=100)
    dataset_prepare(proto_input_path=r'CroppedDataset/B_S/imgs',
                    proto_target_path=r'CroppedDataset/B_S/targets',
                    name="B_S", threshold=100)
    dataset_prepare(proto_input_path=r'CroppedDataset/L_C/imgs',
                    proto_target_path=r'CroppedDataset/L_C/targets',
                    name="L_C", threshold=100)
    dataset_prepare(proto_input_path=r'CroppedDataset/L_S/imgs',
                    proto_target_path=r'CroppedDataset/L_S/targets',
                    name="L_S", threshold=100)
    print("preDataset准备完成")






