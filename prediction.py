from network_2 import VLLSE
from dataloader import get_dataloader_BL
import matplotlib.pyplot as plt
import torch

def predict(paras_path='./model_save/model_paras.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VLLSE(ch_in=1, ch_out=1)
    model.load_state_dict(torch.load(paras_path))
    model.to(device=device)
    model.eval()
    train_dataloader = get_dataloader_BL(mode='train')
    test_dataloader = get_dataloader_BL(mode='test')

    for idx, (imgs,targets) in enumerate(test_dataloader):
        imgs = imgs.to(device=device)
        outputs = model(imgs)
        outputs = outputs>=0.5
        N, C, H, W = outputs.size()

        imgs = imgs.cpu().numpy()
        outputs = outputs.cpu().numpy()
        targets = targets.numpy()

        
        img_list = []
        target_list = []
        output_list = []

        for i in range(N):
            for j in range(C):
                img = imgs[i, j, :, :]
                target = targets[i, j, :, :]
                output = outputs[i, j, :, :]

                img_list.append(img)
                target_list.append(target)
                output_list.append(output)
        
        fig, axes = plt.subplots(3, 5, figsize=(16, 8))
        for i in range(5):
            ax = axes[0, i]
            ax.imshow(img_list[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f'img:{i+1}')
        for i in range(5):
            ax = axes[1, i]
            ax.imshow(output_list[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f'output:{i+1}')
        for i in range(5):
            ax = axes[2, i]
            ax.imshow(target_list[i], cmap='gray')
            ax.axis('off')
            ax.set_title(f'target:{i+1}')
        plt.tight_layout()
        plt.show()

        break
        
if __name__ == "__main__":
    predict()

