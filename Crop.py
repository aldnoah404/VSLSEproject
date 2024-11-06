# 鉴于原始图片较大，为了更好地训练，将图片统一裁剪为288*288像素大小
from PIL import Image
import os
import random
from ToSequence import rm_dir
import re
import shutil
def rm_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path,exist_ok=True)

def make_label(label_path,save_path):
    with open(label_path,encoding='utf_8') as f:
        content = f.readline().strip()
        rm_dir(save_path)
        a = 1
        while content != '':
            label_save_path = os.path.join(save_path,f'{a:0>4}.txt')
            with open(label_save_path,'w',encoding='utf_8') as label:
                content = re.split(' |\t', content)
                x=content[0]
                y=content[1]
                print(x, y, '150.0', '150.0',file=label)
            content = f.readline().strip()
            a = a+1

def crop(proto_imgs_path,proto_labels_path,proto_targets_path,name):
    # 拟输入：
    # proto_imgs_path = r'SequenceDataset\B\imgs'
    # proto_labels_path = r'./SequenceDataset/B/labels'
    # proto_targets_path = r'./SequenceDataset/B/targets'
    # 获取要裁剪的图片与对应特征点位置与标签图片的地址

    rm_dir(f'CroppedDataset/{name}/imgs')
    rm_dir(f'CroppedDataset/{name}/labels')
    rm_dir(f'CroppedDataset/{name}targets')
    imgs_path = proto_imgs_path #r'SequenceDataset\B\imgs'
    labels_path = proto_labels_path
    targets_path = proto_targets_path

    print(f'原始imgs路径：{imgs_path}')
    print(f'原始labels路径：{labels_path}')
    print(f'原始targets路径：{targets_path}')

    # 获取每一张图片的地址并排序
    imgs_sequence_list = [os.path.join(imgs_path, path) for path in os.listdir(imgs_path)] #[r'SequenceDataset\B\imgs\1']
    imgs_sequence_list.sort(key= lambda x :int(x.split('\\')[-1]))

    labels_sequence_list = [os.path.join(labels_path,path) for path in os.listdir(labels_path)]
    labels_sequence_list.sort(key= lambda x :int(x.split('\\')[-1]))

    targets_sequence_list = [os.path.join(targets_path, path) for path in os.listdir(targets_path)]
    targets_sequence_list.sort(key=lambda x: int(x.split('\\')[-1]))

    for i in range(len(imgs_sequence_list)):
        file_name = imgs_sequence_list[i].split('\\')[-1]
        # [r'SequenceDataset\B\imgs\1\1.ipg']
        imgs_list = [os.path.join(imgs_sequence_list[i],x) for x in os.listdir(imgs_sequence_list[i])]
        imgs_list.sort(key= lambda x :int(x.split('\\')[-1][:-4]))
        labels_list = [os.path.join(labels_sequence_list[i],x) for x in os.listdir(labels_sequence_list[i])]
        labels_list.sort(key= lambda x :int(x.split('\\')[-1][:-4]))
        targets_list = [os.path.join(targets_sequence_list[i],x) for x in os.listdir(targets_sequence_list[i])]
        targets_list.sort(key= lambda x :int(x.split('\\')[-1][:-4]))


        # 设置裁剪后图片的保存路径并创建文件夹
        saveimgs_path = f'CroppedDataset/{name}/imgs/{file_name}'
        savelabels_path = f'CroppedDataset/{name}/labels/{file_name}'
        savetargets_path = f'CroppedDataset/{name}/targets/{file_name}'
        rm_dir(saveimgs_path)
        rm_dir(savelabels_path)
        rm_dir(savetargets_path)

        # print(f'裁剪后imgs路径：{saveimgs_path}')
        # print(f'裁剪后labels路径：{savelabels_path}')
        # print(f'裁剪后targets路径：{savetargets_path}')
        
        x_skewing = random.uniform(-5., 5.)
        y_skewing = random.uniform(-5., 5.)

        # 逐一裁剪图片并保存新图片和新label
        for i in range(len(imgs_list)):
            # 读取图片
            
            image = Image.open(imgs_list[i])
            target = Image.open(targets_list[i])
            

            with open(labels_list[i]) as f:

                # 读取标签内容
                text = f.readline().split(' ')
                ori_x = float(text[0])
                ori_y = float(text[1])
                ori_w = float(text[2])
                ori_h = float(text[3])

                # 随机偏移中心一定距离
                
                x_cut = int(ori_x + x_skewing)
                y_cut = int(ori_y + y_skewing)
                left = x_cut - 150
                right = x_cut + 150
                top = y_cut - 150
                bottom = y_cut + 150

                # 裁剪图片
                image = image.crop((left, top, right, bottom))
                target = target.crop((left, top, right, bottom))

                # 保留旧名称并保存图片
                new_name = imgs_list[i].split('\\')[-1]
                new_lname = labels_list[i].split('\\')[-1]
                new_tname = targets_list[i].split('\\')[-1]
                image.save(os.path.join(saveimgs_path,new_name))
                target.save(os.path.join(savetargets_path,new_tname))

                # 计算新的label并保存
                new_x = ori_x - left
                new_y = ori_y - top
                new_w = 150
                new_h = 150
                txt_file = os.path.join(savelabels_path, new_lname)
                with open(txt_file, mode='w') as txt_file:
                    print(new_x, new_y, new_w, new_h, file=txt_file)

    print("————————图片裁剪完成————————\n")


if __name__ == "__main__":

    # make_label(label_path=r'ProtoDataset\scan\B\groundtruth.txt', save_path=r'./ProtoDataset/scan/B/labels')
    # make_label(label_path=r'ProtoDataset\scan\L\groundtruth.txt', save_path=r'./ProtoDataset/scan/L/labels')
    # make_label(label_path=r'ProtoDataset\scan\B_C\groundtruth.txt',save_path=r'./ProtoDataset/scan/B_C/labels')
    # make_label(label_path=r'ProtoDataset\scan\B_S\groundtruth.txt',save_path=r'./ProtoDataset/scan/B_S/labels')
    # make_label(label_path=r'ProtoDataset\scan\L_C\groundtruth.txt',save_path=r'./ProtoDataset/scan/L_C/labels')
    # make_label(label_path=r'ProtoDataset\scan\L_S\groundtruth.txt',save_path=r'./ProtoDataset/scan/L_S/labels')



    # 对B类图片进行裁剪
    crop(proto_imgs_path=r'./SequenceDataset/B/imgs',
         proto_labels_path=r'./SequenceDataset/B/labels',
         proto_targets_path=r'./SequenceDataset/B/targets',
         name='B')
    crop(proto_imgs_path=r'./SequenceDataset/B_C/imgs',
         proto_labels_path=r'./SequenceDataset/B_C/labels',
         proto_targets_path=r'./SequenceDataset/B_C/targets',
         name='B_C')
    crop(proto_imgs_path=r'./SequenceDataset/B_S/imgs',
         proto_labels_path=r'./SequenceDataset/B_S/labels',
         proto_targets_path=r'./SequenceDataset/B_S/targets',
         name='B_S')
    # 对L类图片进行裁剪
    crop(proto_imgs_path=r'./SequenceDataset/L/imgs',
         proto_labels_path=r'./SequenceDataset/L/labels',
         proto_targets_path=r'./SequenceDataset/L/targets',
         name='L')
    crop(proto_imgs_path=r'./SequenceDataset/L_C/imgs',
         proto_labels_path=r'./SequenceDataset/L_C/labels',
         proto_targets_path=r'./SequenceDataset/L_C/targets',
         name='L_C')
    crop(proto_imgs_path=r'./SequenceDataset/L_S/imgs',
         proto_labels_path=r'./SequenceDataset/L_S/labels',
         proto_targets_path=r'./SequenceDataset/L_S/targets',
         name='L_S')