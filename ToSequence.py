# 1.10张图片一组将原始数据集分为序列数据集
# 2.对每一序列的图片及target作相同的transformer，数据增强
# 3.对target做预处理
from PIL import Image
import os
import shutil

def rm_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def ToSequenceDataset(protopath):
    protopath = protopath
    dir_name = protopath.split('/')[-1]
    rm_dir(f'./SequenceDataset/{dir_name}/imgs')
    rm_dir(f'./SequenceDataset/{dir_name}/targets')
    rm_dir(f'./SequenceDataset/{dir_name}/labels')
    input_list = os.listdir(os.path.join(protopath,'noise_imgs'))
    print(f'数据集长度：{len(input_list)}')
    for i in range(0, len(input_list)-9, 5):
        input_save_path = f'./SequenceDataset/{dir_name}/imgs/{i+1}'
        target_save_path = f'./SequenceDataset/{dir_name}/targets/{i+1}'
        label_save_path = f'./SequenceDataset/{dir_name}/labels/{i+1}'
        rm_dir(input_save_path)
        rm_dir(target_save_path)
        rm_dir(label_save_path)
        for j in range(1,11):
            name = f'{i+j}.jpg'
            label = f'{i+j:0>4}.txt'
            input_path = os.path.join(protopath,'noise_imgs', name)
            target_path = os.path.join(protopath,'imgs',name)
            label_path = os.path.join(protopath, 'labels', label)
        
            shutil.copy(input_path,input_save_path)
            shutil.copy(target_path,target_save_path)
            shutil.copy(label_path, label_save_path)

if __name__ == "__main__":
    ToSequenceDataset('./ProtoDataset/scan/B')
    ToSequenceDataset('./ProtoDataset/scan/B_C')
    ToSequenceDataset('./ProtoDataset/scan/B_S')
    ToSequenceDataset('./ProtoDataset/scan/L')
    ToSequenceDataset('./ProtoDataset/scan/L_C')
    ToSequenceDataset('./ProtoDataset/scan/L_S')

