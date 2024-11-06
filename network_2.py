# network_2 包含 SSLSE，unet模型
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=ch_in,out_channels=ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=ch_out,out_channels=ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=ch_in,out_channels=ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        return self.up(x)


class res_conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(res_conv_block,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=ch_in,out_channels=ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=ch_out,out_channels=ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
        )
        self.relu = nn.ReLU()

        self.downsample = None
        if ch_in != ch_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=ch_in,out_channels=ch_out,kernel_size=1,bias=True),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        x1 = self.conv1(x)
        x1 = x1 + identity
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = x2 + x1
        x2 = self.relu(x2)
        return x2


class up(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.up(x)
        return x


class non_local_block(nn.Module):
    def __init__(self, in_channels):
        super(non_local_block, self).__init__()
        self.inter_channels = in_channels // 2
        self.query = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=True)
        self.key = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=True)
        self.value = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, bias=True)
        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        # self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, W, H = x.size()
        proj_query = self.query(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)  # 64
        proj_key = self.key(x).view(batch_size, self.inter_channels, -1)
        proj_value = self.value(x).view(batch_size, self.inter_channels, -1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, W, H)
        out = self.out_conv(out)
        # out = self.gamma * out + x
        out = out + x
        return out


class non_local_conv(nn.Module):
    def __init__(self,ch_in,ch_out):#in 64  out 128
        super(non_local_conv,self).__init__()
        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,padding=1,bias=True)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.non_local_block = non_local_block(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,padding=1,bias=True)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.down_sample = None
        if ch_in != ch_out:
            self.down_sample = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,bias=True),
                nn.BatchNorm2d(ch_out)
            )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self,x):
        identity = x#32 64 72 72
        out = self.conv1(x)#32 128 72 72
        out = self.bn1(out)
        out = self.relu(out)#32 128 72 72
        out = self.non_local_block(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample is not None:
            identity = self.down_sample(x)
        out += identity
        out = self.relu2(out)
        return out


class gate_contr_block(nn.Module):
    def __init__(self,ch_in,ch_out):#512 256
        super(gate_contr_block,self).__init__()
        # self.ein_channels = in_channels//2#256
        self.inter_channels = ch_in//2#256
        self.en_conv = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=2,padding=1,bias=True)
        self.de_conv = nn.Conv2d(ch_in,ch_out,kernel_size=1,bias=True)
        self.va_conv = nn.Conv2d(ch_out,ch_out,kernel_size=1,bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(ch_out,1,kernel_size=1,bias=True)
        self.sigmoid = nn.Sigmoid()
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=1,bias=True)

    def forward(self,en_map,de_map):#256 512
        query = self.de_conv(de_map)#256
        key = self.en_conv(en_map)#256
        value = self.va_conv(en_map)#256
        energy = self.relu(query+key)
        attention = self.conv1(energy)
        attention_weight = self.sigmoid(attention)
        attention_weight = F.interpolate(attention_weight,size=(value.size(2),value.size(3)),mode='bilinear',align_corners=True)
        out = attention_weight * value
        out = self.conv2(out)
        return out


class unet(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(unet,self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.conv1 = conv_block(ch_in,64)
        self.conv2 = conv_block(64,128)
        self.conv3 = conv_block(128,256)
        self.conv4 = conv_block(256,512)
        self.conv5 = conv_block(512,1024)

        self.up5 = up_conv(1024,512)
        self.up_conv5 = conv_block(1024,512)

        self.up4 = up_conv(512,256)
        self.up_conv4 = conv_block(512,256)

        self.up3 = up_conv(256, 128)
        self.up_conv3 = conv_block(256, 128)

        self.up2 = up_conv(128, 64)
        self.up_conv2 = conv_block(128, 64)

        self.conv_1x1 = nn.Conv2d(64,ch_out,kernel_size=1,stride=1,padding=0,bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)

        d5 = self.up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)
        d1 = self.sigmoid(d1)

        return d1


class SSLSE(nn.Module):
    def __init__(self,ch_in=1,ch_out=1):
        super(SSLSE,self).__init__()
        self.Maxpool = nn.MaxPool2d(2)

        self.conv1 = conv_block(ch_in=ch_in,ch_out=32)
        self.res_conv2 = res_conv_block(ch_in=32,ch_out=64)
        self.non_local_conv3 = non_local_conv(ch_in=64,ch_out=128)
        self.non_local_conv4 = non_local_conv(ch_in=128,ch_out=256)
        self.res_conv5 = res_conv_block(ch_in=256,ch_out=512)

        self.up4 = up(ch_in=512,ch_out=256)
        self.gc_conv4 = gate_contr_block(ch_in=512,ch_out=256)
        self.up_conv4 = conv_block(ch_in=512,ch_out=256)

        self.up3 = up(ch_in=256,ch_out=128)
        self.gc_conv3 = gate_contr_block(ch_in=256,ch_out=128)
        self.up_conv3 = conv_block(ch_in=256,ch_out=128)

        self.up2 = up(ch_in=128, ch_out=64)
        self.gc_conv2 = gate_contr_block(ch_in=128, ch_out=64)
        self.up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.up1 = up(ch_in=64, ch_out=32)
        self.gc_conv1 = gate_contr_block(ch_in=64, ch_out=32)
        self.up_conv1 = conv_block(ch_in=64, ch_out=32)

        self.out_conv = nn.Conv2d(in_channels=32,out_channels=ch_out,kernel_size=1,bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x1 = self.conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.res_conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.non_local_conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.non_local_conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.res_conv5(x5)

        d4 = self.up4(x5)
        h4 = self.gc_conv4(x4,x5)
        d4 = torch.cat([d4,h4],dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        h3 = self.gc_conv3(x3,d4)
        d3 = torch.cat([d3,h3],dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        h2 = self.gc_conv2(x2, d3)
        d2 = torch.cat([d2, h2],dim=1)
        d2 = self.up_conv2(d2)

        d1 = self.up1(d2)
        h1 = self.gc_conv1(x1, d2)
        d1 = torch.cat([d1, h1],dim=1)
        d1 = self.up_conv1(d1)

        d0 = self.out_conv(d1)
        out = self.sigmoid(d0)

        return out
    
class conv_GRU(nn.Module):
    def __init__(self, ch_in) -> None:
        super(conv_GRU,self).__init__()
        self.conv_z = nn.Conv2d(in_channels=(ch_in * 2), out_channels=ch_in, kernel_size=3, padding=1, bias=True)
        self.conv_R = nn.Conv2d(in_channels=(ch_in * 2), out_channels=ch_in, kernel_size=3, padding=1, bias=True)
        self.conv_h = nn.Conv2d(in_channels=(ch_in * 2), out_channels=ch_in, kernel_size=3, padding=1, bias=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h_prev, gamma):
        # N, C, H, W = x.size()
        x2 = torch.cat([x, h_prev],dim=1)
        Z = self.conv_z(x2)
        Z = Z * gamma
        Z = self.sigmoid(Z)

        R = self.conv_R(x2)
        R = self.sigmoid(R)

        hidden = torch.cat([(R * h_prev), x],dim=1)
        hidden = self.conv_h(hidden)
        hidden = self.tanh(hidden)

        hidden = Z * h_prev + (1 - Z)*hidden
        return hidden

class VLLSE(nn.Module):
    def __init__(self,ch_in=1,ch_out=1):
        super(VLLSE,self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Maxpool = nn.MaxPool2d(2)

        self.conv1 = conv_block(ch_in=ch_in,ch_out=32)
        self.conv_GRU1 = conv_GRU(ch_in=32)
        self.conv_h1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, padding=0, bias=True)
        self.backconv_h1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0, bias=True)

        self.res_conv2 = res_conv_block(ch_in=32,ch_out=64)
        self.conv_GRU2 = conv_GRU(ch_in=64)
        self.conv_h2 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, padding=0, bias=True)
        self.backconv_h2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding=0, bias=True)

        self.non_local_conv3 = non_local_conv(ch_in=64,ch_out=128)
        self.conv_GRU3 = conv_GRU(ch_in=128)
        self.conv_h3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=1, padding=0, bias=True)
        self.backconv_h3 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, padding=0, bias=True)

        self.non_local_conv4 = non_local_conv(ch_in=128,ch_out=256)
        self.conv_GRU4 = conv_GRU(ch_in=256)
        self.conv_h4 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=1, padding=0, bias=True)
        self.backconv_h4 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, padding=0, bias=True)

        self.res_conv5 = res_conv_block(ch_in=256,ch_out=512)
        self.conv_GRU5 = conv_GRU(ch_in=512)
        self.conv_h5 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=1, padding=0, bias=True)
        self.backconv_h5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, padding=0, bias=True)

        self.up4 = up(ch_in=512,ch_out=256)
        self.gc_conv4 = gate_contr_block(ch_in=512,ch_out=256)
        self.up_conv4 = conv_block(ch_in=512,ch_out=256)

        self.up3 = up(ch_in=256,ch_out=128)
        self.gc_conv3 = gate_contr_block(ch_in=256,ch_out=128)
        self.up_conv3 = conv_block(ch_in=256,ch_out=128)

        self.up2 = up(ch_in=128, ch_out=64)
        self.gc_conv2 = gate_contr_block(ch_in=128, ch_out=64)
        self.up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.up1 = up(ch_in=64, ch_out=32)
        self.gc_conv1 = gate_contr_block(ch_in=64, ch_out=32)
        self.up_conv1 = conv_block(ch_in=64, ch_out=32)

        self.out_conv = nn.Conv2d(in_channels=32,out_channels=ch_out,kernel_size=1,bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x0):
        BS, T, H, W = x0.size()
        out = torch.zeros([BS, T, H, W]).to(device=self.device)
        h_prev1 = torch.zeros([BS, 1, H, W]).to(device=self.device)
        h_prev2 = torch.zeros([BS, 1, int(H/2), int(W/2)]).to(device=self.device)
        h_prev3 = torch.zeros([BS, 1, int(H/4), int(W/4)]).to(device=self.device)
        h_prev4 = torch.zeros([BS, 1, int(H/8), int(W/8)]).to(device=self.device)
        h_prev5 = torch.zeros([BS, 1, int(H/16), int(W/16)]).to(device=self.device)
        gamma = 1
        prev_out = None

        for i in range(T):
            x = x0[:, i, :, :].unsqueeze(dim=1)

            if prev_out != None:
                N_img = ((x*255)>=200).sum()
                N_prev = prev_out.sum()
                drt = N_img - N_prev
                if drt < 10000:
                    gamma = 1
                elif drt <40000:
                    gamma = (drt/30000) + (2/3)
                    # print(f'gamma:{gamma}')

                else:
                    gamma = 2
                    # print(f'gamma:{gamma}')
            # print(gamma)

            x1 = self.conv1(x)
            h1 = self.conv_h1(h_prev1)
            x1 = self.conv_GRU1(x1, h1, gamma=gamma)
            h_prev1 = self.backconv_h1(x1).detach().clone()  

            x2 = self.Maxpool(x1)
            x2 = self.res_conv2(x2)
            h2 = self.conv_h2(h_prev2)
            x2 = self.conv_GRU2(x2, h2, gamma=gamma)
            h_prev2 = self.backconv_h2(x2).detach().clone()  

            x3 = self.Maxpool(x2)
            x3 = self.non_local_conv3(x3)
            h3 = self.conv_h3(h_prev3)
            x3 = self.conv_GRU3(x3, h3, gamma=gamma)
            h_prev3 = self.backconv_h3(x3).detach().clone()  

            x4 = self.Maxpool(x3)
            x4 = self.non_local_conv4(x4)
            h4 = self.conv_h4(h_prev4)
            x4 = self.conv_GRU4(x4, h4, gamma=gamma)
            h_prev4 = self.backconv_h4(x4).detach().clone()  

            x5 = self.Maxpool(x4)
            x5 = self.res_conv5(x5)
            h5 = self.conv_h5(h_prev5)
            x5 = self.conv_GRU5(x5, h5, gamma=gamma)
            h_prev5 = self.backconv_h5(x5).detach().clone()  

            # h_prev1[:, i, :, :] = h1.squeeze(dim=1).detach().clone()  
            # h_prev2[:, i, :, :] = h2.squeeze(dim=1).detach().clone()  
            # h_prev3[:, i, :, :] = h3.squeeze(dim=1).detach().clone()  
            # h_prev4[:, i, :, :] = h4.squeeze(dim=1).detach().clone()  
            # h_prev5[:, i, :, :] = h5.squeeze(dim=1).detach().clone()  

            d4 = self.up4(x5)
            hh4 = self.gc_conv4(x4,x5)
            d4 = torch.cat([d4,hh4],dim=1)
            d4 = self.up_conv4(d4)

            d3 = self.up3(d4)
            hh3 = self.gc_conv3(x3,d4)
            d3 = torch.cat([d3,hh3],dim=1)
            d3 = self.up_conv3(d3)

            d2 = self.up2(d3)
            hh2 = self.gc_conv2(x2, d3)
            d2 = torch.cat([d2, hh2],dim=1)
            d2 = self.up_conv2(d2)

            d1 = self.up1(d2)
            hh1 = self.gc_conv1(x1, d2)
            d1 = torch.cat([d1, hh1],dim=1)
            d1 = self.up_conv1(d1)

            d0 = self.out_conv(d1)
            d0 = self.sigmoid(d0)
            prev_out = (d0>=0.5).float()
            out[:, i, :, :] = d0.squeeze(dim=1)

        return out


if __name__ == '__main__':
    # model = unet(ch_in=1,ch_out=1)
    # # model = SSLSE(ch_in=1,ch_out=1)
    # params = list(model.parameters())
    # k = 0
    # for i in params:
    #     l = 1
    #     # print("该层的结构：" + str(list(i.size())))
    #     for j in i.size():
    #         l *= j
    #     # print("该层参数和：" + str(l))
    #     k = k + l
    # print("总参数数量和：" + str(k))
    model = VLLSE(ch_in=1,ch_out=1)
    x0 = torch.rand([2,10,288,288])
    h_prev1 = torch.zeros([2,10,288,288])
    h_prev2 = torch.zeros([2,10,144,144])
    h_prev3 = torch.zeros([2,10,72,72])
    h_prev4 = torch.zeros([2,10,36,36])
    h_prev5 = torch.zeros([2,10,18,18])

    out,h1,h2,h3,h4,h5=model.forward(x0=x0,h_prev1=h_prev1,h_prev2=h_prev2,h_prev3=h_prev3,h_prev4=h_prev4,h_prev5=h_prev5)
    print(out.size())
    print(h1.size())
    print(h2.size())
    print(h3.size())
    print(h4.size())
    print(h5.size())

    print(h1[1,0,1,1])











