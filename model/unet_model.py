from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torch.nn.parameter import Parameter
from model.GEblock import GEblock
import numpy as np

disp = False

class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out

class ln(nn.Module):
    """
    Layer Normalization
    """
    def __init__(self, input):
        super(ln, self).__init__()
        self.ln = nn.LayerNorm(input.size()[1:]).cuda()
    def forward(self, x):
        x = self.ln(x)
        return x

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch, down=False):
        super(conv_block, self).__init__()
        if down:
            stride = 2
        else:
            stride = 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.SELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.SELU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class Conv_block(nn.Module):
    """
    Global-Feature Encoding U-Net Convolution Block
    """
    def __init__(self, in_ch, out_ch, kn_size, stride, t=2, pad_mode='SYMMETRIC', Normalize=True, Selu=True):
        super(Conv_block, self).__init__()
        self.pad_size = (kn_size - 1) // 2
        self.pad_mode = pad_mode
        self.Normalize = Normalize
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kn_size = kn_size
        self.stride = stride
        self.Selu = Selu
        self.t = t

        self.conv_block1 = self._make_layer1()
        self.conv_block2 = self._make_layer2()

    def _make_layer1(self):
        layers = []
        if self.pad_size > 0 and self.pad_mode is not None:
            layers.append(nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.kn_size, stride=self.stride, padding=self.pad_size, padding_mode=self.pad_mode))
        else:
            layers.append(nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.kn_size, stride=self.stride))
        if self.Normalize == True:
            layers.append(nn.InstanceNorm2d(self.out_ch,affine=True,track_running_stats=True))
        if self.Selu == True:
            layers.append(nn.SELU(inplace=True))
        return nn.Sequential(*layers)

    def _make_layer2(self):
        layers = []
        if self.pad_size > 0 and self.pad_mode is not None:
            layers.append(nn.Conv2d(self.out_ch, self.out_ch, kernel_size=self.kn_size, stride=self.stride, padding=self.pad_size, padding_mode=self.pad_mode))
        else:
            layers.append(nn.Conv2d(self.out_ch, self.out_ch, kernel_size=self.kn_size, stride=self.stride))
        if self.Normalize == True:
            layers.append(nn.InstanceNorm2d(self.out_ch,affine=True,track_running_stats=True))
        if self.Selu == True:
            layers.append(nn.SELU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.t == 0 :
            x1 = self.conv_block1(x)
        else:
            for i in range(self.t):
                if i == 0:
                    x1 = self.conv_block2(x)
                x1 = self.conv_block2(x+x1)
        if disp:
            print('conv', x1.shape)
        return x1


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.In = nn.InstanceNorm2d(out_ch,affine=True,track_running_stats=True)
        self.selu = nn.SELU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x = F.interpolate(x, y.size()[2:], mode='bilinear', align_corners=True)
        z = self.conv(x)
        z = self.bn(z)
        z = self.relu(z)
        return z

class res_layer(nn.Module):
    """
    Resblock layer
    """

    def __init__(self):
        super(res_layer, self).__init__()

    def forward(self, x, y, axis):
        if axis is not None:
            l = [y[:, i, :, :] for i in axis]
            y = torch.stack(l, 1)
        tensor = torch.add(x, y)
        return tensor

class Global_concat_layer(nn.Module):
    """
    Global concat layer
    """
    def __init__(self):
        super(Global_concat_layer, self).__init__()

    def tile(self, a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index.cuda())

    def forward(self, x, y):
        batch_size = x.size()[0]
        h = x.size()[2]
        w = x.size()[3]
        concat_t = torch.squeeze(y, dim=2)
        concat_t = torch.squeeze(concat_t, dim=2)
        dims = concat_t.size()[-1]
        concat_t = torch.reshape(concat_t, [batch_size, dims])
        bs = []
        for i in range(batch_size):
            batch = concat_t[i, :]
            batch = self.tile(batch, 0, h * w)
            batch = torch.reshape(batch, [-1, h, w])
            bs.append(batch)
        concat_t = torch.stack(bs)
        tensor = torch.cat([x, concat_t], 1)
        return tensor

class Global_extract_layer(nn.Module):
    def __init__(self, channels=2048):
        """
        Global Feature Pyramid Extraction Module
        :type channels: int
        """
        super(Global_extract_layer, self).__init__()
        channels_mid = int(channels)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=3, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)
        self.conv_master2 = nn.Conv2d(self.channels_cond * 2, self.channels_cond, kernel_size=3,bias=False)
        self.bn_master2 = nn.BatchNorm2d(channels)

        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.global_concat_layer = Global_concat_layer()
        self.conv7x7_2 = GEblock(channels_mid, 16,'att')
        self.conv5x5_2 = GEblock(channels_mid, 16,'att')
        self.conv3x3_2 = GEblock(channels_mid, 16,'att')

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, 2048, h, w]
        :return: out: Feature maps. Shape: [b, 2048, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Branch 1
        x1_1 = self.conv3x3_1(x)
        x1_1 = self.bn3_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv3x3_2(x1_1)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)

        # Branch 3
        x3_1 = self.conv7x7_1(x2_1)
        x3_1 = self.bn1_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv7x7_2(x3_1)

        # Merge branch 1 and 2
        g1 = x1_2 + x2_2 + x3_2
        g2 = self.global_concat_layer(x_master,g1)
        g3 = self.conv_master2(g2)
        g3 = self.bn_master2(g3)
        out = self.relu(g3)

        return out


class GAU(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True):
        super(GAU, self).__init__()
        """
        Global Attention Connection Upsample module
        """
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_upsample = nn.BatchNorm2d(channels_low)

            self.conv_upsample2 = nn.Conv2d(channels_low*2, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_upsample2 = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=3, padding=1, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        """
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :return: fms_att_upsample
        """
        b, c, h, w = fms_high.shape

        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        if self.upsample:
            high = self.relu(
                self.bn_upsample(self.conv_upsample(nn.Upsample(size=fms_att.size()[2:],mode='bilinear',align_corners=True)(fms_high))))
            out = torch.cat((high, fms_att), dim=1)
            out = self.relu(self.bn_upsample2(self.conv_upsample2(out)))
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)
        return out


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5,e4)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5,e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4,e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3,e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        # d1 = self.active(out)

        return out


class GEU_Net(nn.Module):
    """
    The proposed GEU-Net network
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(GEU_Net, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1],down=True)
        self.Conv3 = conv_block(filters[1], filters[2],down=True)
        self.Conv4 = conv_block(filters[2], filters[3],down=True)
        self.Conv5 = conv_block(filters[3], filters[4],down=True)

        self.GE_layer = Global_extract_layer(filters[4])

        self.gau_block1 = GAU(filters[4], filters[3])
        self.gau_block2 = GAU(filters[3], filters[2])
        self.gau_block3 = GAU(filters[2], filters[1])
        self.gau_block4 = GAU(filters[1], filters[0])
        self.gau = [self.gau_block1, self.gau_block2, self.gau_block3, self.gau_block4]

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Out1 = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.Out2 = res_layer()
    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        # Downsampling
        e1 = self.Conv1(x)
        e2 = self.Conv2(e1)
        e3 = self.Conv3(e2)
        e4 = self.Conv4(e3)
        e5 = self.Conv5(e4)
        # Extract global features
        g1 = self.GE_layer(e5)
        # Upsampling
        d5 = self.gau[0](g1, e4)
        x5 = self.Up5(g1, e4)
        d5 = torch.cat((x5, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.gau[1](d5, e3)
        x4 = self.Up4(d5, e3)
        d4 = torch.cat((d4, x4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.gau[2](d4, e2)
        x3 = self.Up3(d4, e2)
        d3 = torch.cat((d3, x3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.gau[3](d3, e1)
        x2 = self.Up2(d3, e1)
        d2 = torch.cat((d2, x2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Out1(d2)
        out = self.Out2(out,x,[0,1])

        # d1 = self.active(out)

        return out


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """
    def __init__(self, in_ch, out_ch, t=2, down=False):
        super(RRCNN_block, self).__init__()
        if down:
            stride = 2
        else:
            stride = 1
        self.RCNN = nn.Sequential(
            Conv_block(out_ch,out_ch, kn_size=3,stride=1,t=t),
            Conv_block(out_ch,out_ch, kn_size=3,stride=1,t=t)
        )
        self.conv1 = Conv_block(in_ch,out_ch, kn_size=1,stride=stride,t=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out


class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """

    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        # out = self.active(out)

        return out


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int,affine=True,track_running_stats=True)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int,affine=True,track_running_stats=True)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1,affine=True,track_running_stats=True),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=3, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # print(x5.shape)
        d5 = self.Up5(e5,e4)
        # print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5,e3)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4,e2)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3,e1)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out


class R2AttU_Net(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """

    def __init__(self, in_ch=3, out_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNN_block(in_ch, filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t, down=True)
        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t, down=True)
        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t, down=True)
        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t, down=True)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.RRCNN1(x)

        e2 = self.RRCNN2(e1)

        e3 = self.RRCNN3(e2)

        e4 = self.RRCNN4(e3)

        e5 = self.RRCNN5(e4)

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out
