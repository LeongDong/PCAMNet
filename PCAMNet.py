import torch
from torch import nn
import torch.nn.functional as F

class ChanCom(nn.Module):
    def __init__(self,in_ch):
        super(ChanCom,self).__init__()
        self.compress = nn.Conv3d(in_ch,1,kernel_size=1)

    def forward(self, x):
        outx = self.compress(x)
        return outx

class SideUp(nn.Module):
    def __init__(self,scale):
        super(SideUp,self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale,mode='trilinear',align_corners=True)

    def forward(self,x):
        outx = self.upsample(x)
        return outx

def soft_dilate(img):
    return F.max_pool3d(img,(5,5,5),(1,1,1),(2,2,2))

def soft_erode(img):
    p1 = -F.max_pool3d(-img,(5,1,1),(1,1,1),(2,0,0))
    p2 = -F.max_pool3d(-img,(1,5,1),(1,1,1),(0,2,0))
    p3 = -F.max_pool3d(-img,(1,1,5),(1,1,1),(0,0,2))

    return torch.min(torch.min(p1,p2),p3)
class clus_atten(nn.Module):
    def __init__(self):
        super(clus_atten,self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.k = nn.Parameter(torch.ones(1))

    def forward(self, feat_area, fore_mask):

        dilate_mask = fore_mask
        erode_mask = fore_mask
        N,C,H,W,S = feat_area.size()
        iters = 1

        for i in range(iters):
            dilate_mask = soft_dilate(fore_mask)
        for i in range(iters):
            erode_mask = soft_erode(fore_mask)

        fore_mask = erode_mask #N,1,H,W,S cand_mask *
        back_mask = 1 - dilate_mask #N,1,H,W,S
        #feat_area = feat_area * cand_mask #N,C,H,W,S

        fore_feat = fore_mask.contiguous().view(N,1,-1) #N,1,HWS
        fore_feat = fore_feat.permute(0,2,1).contiguous() #N,HWS,1
        back_feat = back_mask.contiguous().view(N,1,-1) #N,1,HWS
        back_feat = back_feat.permute(0,2,1).contiguous() #N,HWS,1
        feat = feat_area.contiguous().view(N,C,-1) #N,C,HWS

        fore_num = torch.sum(fore_feat,dim=1,keepdim=True) + 1e-5
        back_num = torch.sum(back_feat,dim=1,keepdim=True) + 1e-5

        fore_cluster = torch.bmm(feat,fore_feat) / fore_num #N,C,1
        back_cluster = torch.bmm(feat,back_feat) / back_num #N,C,1
        feat_cluster = torch.cat((fore_cluster,back_cluster),dim=-1) #N,C,2

        feat_key = feat_area #N,C,H,W,S
        feat_key = feat_key.contiguous().view(N,C,-1) #N,C,HWS
        feat_key = feat_key.permute(0,2,1).contiguous() #N,HWS,C

        feat_cluster = feat_cluster.permute(0,2,1).contiguous() #N,2,C
        feat_query = feat_cluster #N,2,C
        feat_value = feat_cluster #N,2,C

        feat_query = feat_query.permute(0,2,1).contiguous() #N,C,2
        feat_sim = torch.bmm(feat_key,feat_query) #N,HWS,2
        feat_sim = self.softmax(feat_sim)

        feat_atten = torch.bmm(feat_sim,feat_value) #N,HWS,C
        feat_atten = feat_atten.permute(0,2,1).contiguous() #N,C,HWS
        feat_atten = feat_atten.view(N,C,H,W,S)
        feat_area = self.k * feat_atten + feat_area

        return feat_area

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.LeakyReLU(negative_slope=0.01,inplace=False))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x

class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.LeakyReLU(negative_slope=0.01,inplace=False))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.LeakyReLU(negative_slope=0.01,inplace=False))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.LeakyReLU(negative_slope=0.01,inplace=False))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class PCAMNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, n_filters=16, normalization='instancenorm', has_dropout=False):
        super(PCAMNet, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.chancom_six = ChanCom(n_filters * 8)
        self.sideup_six = SideUp(8)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.chancom_seven = ChanCom(n_filters * 4)
        self.sideup_seven = SideUp(4)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.chancom_eight = ChanCom(n_filters * 2)
        self.sideup_eight = SideUp(2)
        self.atten = clus_atten()
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)
        # self.__init_weight()

    def encoder(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

    def decoder(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_side = self.chancom_six(x6)
        x6_sideout = self.sideup_six(x6_side)
        # x6_mask = self.sigmoid(x6_side)
        # x6_atten = self.atten(x6,x6_mask)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_side = self.chancom_seven(x7)
        x7_sideout = self.sideup_seven(x7_side)
        # x7_mask = self.sigmoid(x7_side)
        # x7_atten = self.atten(x7,x7_mask)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_side = self.chancom_eight(x8)
        x8_sideout = self.sideup_eight(x8_side)
        x8_mask = self.sigmoid(x8_side)
        x8_atten = self.atten(x8,x8_mask)
        x8_up = self.block_eight_up(x8_atten)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        result = [x6_sideout, x7_sideout, x8_sideout, out]
        out_seg = [self.sigmoid(r) for r in result]
        return out_seg


    def forward(self, input, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features = self.encoder(input)
        out_seg = self.decoder(features)
        if turnoff_drop:
            self.has_dropout = has_dropout
        return out_seg
