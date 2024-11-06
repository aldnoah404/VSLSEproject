import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,ConcatDataset
import os
import random
import shutil
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import functional

# 训练集应作一定处理
class make_dataset(Dataset):
    def __init__(self,protopath, mode='train', augmentation_prob=0.4) -> None:
        super(make_dataset,self).__init__()
        # 拟输入：protopath = './Datasets/train/B'
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.protopath = protopath
        self.imgs_path = os.path.join(protopath,'inputs') # './Datasets/train/B/inputs'
        self.imgs_list = os.listdir(self.imgs_path) # '6'
        self.imgs_list.sort(key=lambda x : int(x))
        self.targets_path = os.path.join(protopath,'targets') # './Datasets/train/B/targets'
        self.targets_list = os.listdir(self.targets_path) # '4'
        self.targets_list.sort(key=lambda x : int(x))
        self.RotationDegree = [0, 90, 180, 270]

    def __getitem__(self, index) -> any:
        # name = self.imgs_list[index]
        input_dir_path = os.path.join(self.imgs_path,self.imgs_list[index]) # './Datasets/train/B/inputs/1'
        input_list = os.listdir(input_dir_path) # [1.jpg, ......]
        input_list.sort(key = lambda x : int(x[:-4]))
        target_dir_path = os.path.join(self.targets_path,self.targets_list[index]) # './Datasets/train/B/targets/1'
        target_list = os.listdir(target_dir_path) # [1.jpg, ......]
        target_list.sort(key = lambda x : int(x[:-4]))
        p_transform = random.random()
        Transform = []
        imgs = []
        gts = []
        T = len(input_list)
        for i in range(T):
            input_path = os.path.join(input_dir_path, input_list[i]) # './Datasets/train/B/inputs/1/1.jpg'
            img = Image.open(input_path)
            target_path = os.path.join(target_dir_path, target_list[i])
            tgt = Image.open(target_path)
            imgs.append(img)
            gts.append(tgt)
                
        if (self.mode=='train') and (p_transform <= self.augmentation_prob):
            RotationDegree = random.randint(0,3)
            RotationDegree = self.RotationDegree[RotationDegree]
            Transform.append(transforms.RandomRotation((RotationDegree,RotationDegree)))
            RotationDegree_2 = random.randint(-10,10)
            Transform.append(transforms.RandomRotation((RotationDegree_2,RotationDegree_2)))
            Transform.append(transforms.CenterCrop(288))
            Transform =transforms.Compose(Transform)
            for i in range(T):
                imgs[i] = Transform(imgs[i])
                gts[i] = Transform(gts[i])
            if random.random() <= 0.5:
                for j in range(T):
                    imgs[j] = functional.hflip(imgs[j])
                    gts[j] = functional.hflip(gts[j])
            if random.random() <= 0.5:
                for j in range(T):
                    imgs[j] = functional.hflip(imgs[j])
                    gts[j] = functional.hflip(gts[j])
            Transform = []
        Transform.append(transforms.CenterCrop(288))
        Transform.append(transforms.ToTensor())
        Transform = transforms.Compose(Transform)
        for i in range(T):
            imgs[i] = Transform(imgs[i])
            gts[i] = Transform(gts[i])

        input = torch.rand(T, 288, 288)
        for i in range(T):
            image = imgs[i]
            input[i, :, :] = image
        target = torch.rand(T, 288, 288)
        for i in range(T):
            gt = gts[i]
            target[i, :, :] = gt
        return input,target
    
    def __len__(self):
        return len(self.imgs_list)
    
# 将数据集划分为训练集和测试集
def split_dataset(datasets_path='./preDataset',p=0.7):
    names = os.listdir(datasets_path)
    for name in names:
        imgs_path = os.path.join(datasets_path,name,'inputs')
        targets_path = os.path.join(datasets_path,name,'targets') 
        imgs_list = os.listdir(imgs_path)
        targets_list = os.listdir(targets_path)
        imgs_list.sort(key = lambda x : int(x))
        targets_list.sort(key = lambda x : int(x))

        train_len = int(len(imgs_list) * p)
        # test_len = int(len(imgs_list) - train_len)

        random_seed = random.random()
        random.seed(random_seed)
        random.shuffle(imgs_list)
        random.shuffle(targets_list)
        
        train_imgs = imgs_list[:train_len]
        test_imgs = imgs_list[train_len:]
        train_targets = targets_list[:train_len]
        test_targets = targets_list[train_len:]

        train_save_path = f'Datasets/train/{name}'
        test_save_path = f'Datasets/test/{name}'
        rm_dir(train_save_path)
        rm_dir(test_save_path)

        for img in train_imgs:
            img_path = os.path.join(imgs_path,img)
            target_path = os.path.join(targets_path,img)
            img_save_path = os.path.join(train_save_path,'inputs',img)
            target_save_path = os.path.join(train_save_path,'targets',img)
            rm_dir(img_save_path)
            rm_dir(target_save_path)
            shutil.copytree(img_path,img_save_path,dirs_exist_ok=True)
            shutil.copytree(target_path,target_save_path,dirs_exist_ok=True)
        for img in test_imgs:
            img_path = os.path.join(imgs_path,img)
            target_path = os.path.join(targets_path,img)
            img_save_path = os.path.join(test_save_path,'inputs',img)
            target_save_path = os.path.join(test_save_path,'targets',img)
            rm_dir(img_save_path)
            rm_dir(target_save_path)
            shutil.copytree(img_path,img_save_path,dirs_exist_ok=True)
            shutil.copytree(target_path,target_save_path,dirs_exist_ok=True)


def rm_dir(dir_path):
    if os.path.exists(dir_path):
        print(f'remove_path: {dir_path}')
        shutil.rmtree(dir_path)
    print(f'recreat_path: {dir_path}')
    os.makedirs(dir_path)


def get_dataloader(test_dataset_path='./Datasets/test',train_dataset_path='./Datasets/train',mode='train'):
    test_dataset_path = test_dataset_path # './Datasets/test'
    train_dataset_path = train_dataset_path
    if mode == 'train':
        train_dataset = None
        for name in os.listdir(train_dataset_path): # name: B
            dir_path = os.path.join(train_dataset_path,name) # './Datasets/test/B'
            dataset = make_dataset(dir_path,mode='train')
            if train_dataset == None:
                train_dataset = dataset
            else:
                train_dataset += dataset
        train_dataloader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=2,drop_last=True, num_workers=2)
        print('train_dataloader加载完成')
        return train_dataloader
    if mode == 'test':
        test_dataset = None
        for name in os.listdir(test_dataset_path):
            dir_path = os.path.join(test_dataset_path,name)
            dataset = make_dataset(dir_path, mode='test')
            if test_dataset == None:
                test_dataset = dataset
            else:
                test_dataset += dataset
        test_dataloader = DataLoader(dataset=test_dataset,shuffle=True,batch_size=2,drop_last=True, num_workers=2)
        print('test_dataloader加载完成')
        return test_dataloader

def get_dataloader_BL(test_dataset_path='./Datasets/test',train_dataset_path='./Datasets/train',mode='train'):
    test_dataset_path = test_dataset_path # './Datasets/test'
    train_dataset_path = train_dataset_path
    if mode == 'train':
        train_dataset = None
        for name in os.listdir(train_dataset_path): # name: B
            if len(name) == 1:
                dir_path = os.path.join(train_dataset_path,name) # './Datasets/test/B'
                dataset = make_dataset(dir_path,mode='train')
                if train_dataset == None:
                    train_dataset = dataset
                else:
                    train_dataset += dataset
        train_dataloader = DataLoader(dataset=train_dataset,shuffle=True,batch_size=2)
        print(f'train数据集长度：{len(train_dataset)}')
        print('train_dataloader加载完成\n')
        return train_dataloader
    if mode == 'test':
        test_dataset = None
        for name in os.listdir(test_dataset_path):
            if len(name) == 1:
                dir_path = os.path.join(test_dataset_path,name)
                dataset = make_dataset(dir_path, mode='test')
                if test_dataset == None:
                    test_dataset = dataset
                else:
                    test_dataset += dataset
        test_dataloader = DataLoader(dataset=test_dataset,shuffle=True,batch_size=2)
        print(f'test数据集长度：{len(test_dataset)}')
        print('test_dataloader加载完成\n')
        return test_dataloader

if __name__ == "__main__":
    
    # split_dataset(datasets_path='./preDataset')

    train_dataloader = get_dataloader_BL(mode='train')
    for idx,(imgs,targets) in enumerate(train_dataloader):
        print(imgs.shape)
    test_dataloader = get_dataloader_BL(mode='test')