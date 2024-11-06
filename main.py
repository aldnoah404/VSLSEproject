import torch
import os
from thop import profile
from network_2 import VLLSE
from dataloader import get_dataloader_BL,rm_dir
from DicexFocal_Loss import DiceFocal_Loss
from torch.optim import Adam, lr_scheduler
from evaluation import *


def main():

    # 设置device并打印信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'——————>device_info：{device}')

    # 实例化模型并打印信息
    model = VLLSE(ch_in=1, ch_out=1)
    model.to(device=device)

    print('——————>model info: VVLSE')
    input1 = torch.randn(2, 10, 288, 288) 
    input1 = input1.to(device=device)
    flops, params = profile(model, inputs=(input1, ))
    print(f'——————>FLOPs = {flops / 1000 ** 3:.3f} G')
    print(f'——————>params = {params / 1000 ** 2:.3f} M')

    # 实例化损失函数
    criterion = DiceFocal_Loss()
    criterion.to(device=device) 

    # 实例化Adam优化器，并设置学习率策略
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=64, eta_min=1e-5)

    # 加载数据加载器，并打印数据集长度
    train_dataloader = get_dataloader_BL(mode='train')
    test_dataloader = get_dataloader_BL(mode='test')

    # 设置模型保存路径
    save_file = r'./model_save'
    save_paras = r'./model_save/model_paras.pth'
    if os.path.exists(save_paras):
        ans = input("********是否重新开始训练？（y/n）:")
        if ans == 'y' or ans == 'Y':
            rm_dir(save_file)
            print()
        elif ans == 'n' or ans == 'N':
            model.load_state_dict(torch.load(save_paras))
            print(f'————模型加载完成')
        else:
            print('————输入错误\n')
    else:
        rm_dir(save_file)

    # 开始训练
    model.train()
    print('————————>train_start<————————')

    epochs = 200
    print(f'epochs: {epochs}')
    flag = 0
    epoch_flag = None

    for epoch in range(epochs):
        print(f'————>epoch: {epoch+1}<————')

        for idx, (imgs, targets) in enumerate(train_dataloader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            print(f'loss:{loss}')
            loss.backward()

            # 应用基于 L2 范数的梯度裁剪  
            max_norm = 1.0  # 设置最大范数  
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
             
            optimizer.step()
            
            if (idx + 1) % 25 == 0:
                acc = get_accuracy(outputs, targets)
                dice = get_dice(outputs, targets)
                dice2 = get_dice2(outputs, targets)
                iou = get_iou(outputs, targets)
                precession = get_precision(outputs, targets)
                recall = get_recall(outputs, targets)
                result_loss = loss.item()
                print(f'idx：{idx + 1} ————>loss  :{result_loss:.5f}\n\t'
                        f'————>acc   :{acc:.5f}\n\t'
                        f'————>dice  :{dice:.5f}\n\t'
                        f'————>dice2 :{dice2:.5f}\n\t'
                        f'————>iou   :{iou:.5f}\n\t'
                        f'————>pc    :{precession:.5f}\n\t'
                        f'————>rc    :{recall:.5f}')

        scheduler.step()

        # if (epoch+1) % 10 == 0:
        save_path = f'model_save/epoch{epoch+1}_paras.pth'
        torch.save(model.state_dict(), save_path)
        print(f'epoch{epoch+1}的参数已保存')

        # 测试性能
        model.eval()
        accs = []
        dices = []
        dice2s = []
        ious = []
        precessions = []
        recalls = []
        losses = []
        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(test_dataloader):
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)

                targets = targets.cpu()
                outputs = outputs.cpu()

                result_acc = get_accuracy(output=outputs, target=targets)
                result_dice = get_dice(output=outputs, target=targets)
                result_dice2 = get_dice2(outputs, targets)
                result_iou = get_iou(outputs, targets)
                result_precession = get_precision(outputs, targets)
                result_recall = get_recall(outputs, targets)
                result_loss = criterion(outputs, targets).item()

                losses.append(result_loss)
                accs.append(result_acc)
                dices.append(result_dice)
                dice2s.append(result_dice2)
                ious.append(result_iou)
                precessions.append(result_precession)
                recalls.append(result_recall)

            acc = np.array(accs).mean()
            dice = np.array(dices).mean()
            dice2 = np.array(dice2s).mean()
            precession = np.array(precessions).mean()
            recall = np.array(recalls).mean()
            iou = np.array(ious).mean()
            loss = np.array(losses).mean()

            print(f'————————test————————')
            print(f' ————>loss  :{loss:.5f}\n\t'
                        f'————>acc   :{acc:.5f}\n\t'
                        f'————>dice  :{dice:.5f}\n\t'
                        f'————>dice2 :{dice2:.5f}\n\t'
                        f'————>iou   :{iou:.5f}\n\t'
                        f'————>pc    :{precession:.5f}\n\t'
                        f'————>rc    :{recall:.5f}')

            if (acc+dice) > flag:
                flag = (acc+dice)
                epoch_flag = epoch+1
                # save_path = f'./model_save/model_paras.pth'
                save_path = f'model_save/model_paras.pth'

                torch.save(model.state_dict(), save_path)
            print('model_paras.pth参数已更新')
            print(f'epoch_flag:{epoch_flag}\n')

if __name__ == "__main__":
    main()
