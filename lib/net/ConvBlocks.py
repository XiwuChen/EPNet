import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import grid_sample

BatchNorm2d = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Conv7x7Block(nn.Module):
    def __init__(self, inplanes, outplanes, stride=2, use_relu=False):
        super(Conv7x7Block, self).__init__()
        if use_relu:
            self.seq = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=7, stride=stride,
                                               padding=3, bias=False),
                                     BatchNorm2d(outplanes),
                                     nn.ReLU(inplace=True),
                                     )
        else:
            self.seq = nn.Sequential(nn.Conv2d(inplanes, outplanes, kernel_size=7, stride=stride,
                                               padding=3, bias=False),
                                     )

    def forward(self, x):
        out = self.seq(x)
        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride = 1, use_relu=False, layer_nums=2):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, outplanes, stride)
        self.bn1 = BatchNorm2d(outplanes )
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(outplanes, outplanes, 2*stride)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        return out
# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, outplanes, stride=1, use_relu=False, layer_nums=2):
#         super(BasicBlock, self).__init__()
#         if use_relu:
#             self.seq = nn.Sequential(conv3x3(inplanes, outplanes, stride), BatchNorm2d(outplanes),
#                                      nn.ReLU(inplace=True),
#                                      conv3x3(outplanes, outplanes, 2 * stride), BatchNorm2d(outplanes),
#                                      nn.ReLU(inplace=True))
#         else:
#             if layer_nums == 2:
#                 self.seq = nn.Sequential(conv3x3(inplanes, outplanes, stride), BatchNorm2d(outplanes),
#                                          nn.ReLU(inplace=True),
#                                          conv3x3(outplanes, outplanes, 2 * stride)
#                                          )
#             else:
#                 self.seq = nn.Sequential(conv3x3(inplanes, outplanes, stride), BatchNorm2d(outplanes),
#                                          nn.ReLU(inplace=True),
#                                          nn.AvgPool2d(kernel_size=2)
#                                          )
#
#     def forward(self, x):
#         out = self.seq(x)
#         return out


class Fusion_Conv(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Fusion_Conv, self).__init__()

        self.conv1 = torch.nn.Conv1d(inplanes, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


# ================addition attention (add)=======================#
class IA_Layer(nn.Module):
    def __init__(self, channels):
        print('##############ADDITION ATTENTION(ADD)#########')
        super(IA_Layer, self).__init__()
        self.ic, self.pc = channels
        rc = self.pc // 4
        self.conv1 = nn.Sequential(nn.Conv1d(self.ic, self.pc, 1),
                                   nn.BatchNorm1d(self.pc),
                                   nn.ReLU())
        self.fc1 = nn.Linear(self.ic, rc)
        self.fc2 = nn.Linear(self.pc, rc)
        self.fc3 = nn.Linear(rc, 1)

    def forward(self, img_feas, point_feas):
        batch = img_feas.size(0)
        img_feas_f = img_feas.transpose(1, 2).contiguous().view(-1, self.ic)  # BCN->BNC->(BN)C
        point_feas_f = point_feas.transpose(1, 2).contiguous().view(-1, self.pc)  # BCN->BNC->(BN)C'
        # print(img_feas)
        ri = self.fc1(img_feas_f)
        rp = self.fc2(point_feas_f)
        att = F.sigmoid(self.fc3(F.tanh(ri + rp)))  # BNx1
        att = att.squeeze(1)
        att = att.view(batch, 1, -1)  # B1N
        # print(img_feas.size(), att.size())

        img_feas_new = self.conv1(img_feas)
        out = img_feas_new * att

        return out


class Atten_Fusion_Conv(nn.Module):
    def __init__(self, inplanes_I, inplanes_P, outplanes):
        super(Atten_Fusion_Conv, self).__init__()

        self.IA_Layer = IA_Layer(channels=[inplanes_I, inplanes_P])
        # self.conv1 = torch.nn.Conv1d(inplanes_P, outplanes, 1)
        self.conv1 = torch.nn.Conv1d(inplanes_P + inplanes_P, outplanes, 1)
        self.bn1 = torch.nn.BatchNorm1d(outplanes)

    def forward(self, point_features, img_features):
        # print(point_features.shape, img_features.shape)

        img_features = self.IA_Layer(img_features, point_features)
        # print("img_features:", img_features.shape)

        # fusion_features = img_features + point_features
        fusion_features = torch.cat([point_features, img_features], dim=1)
        fusion_features = F.relu(self.bn1(self.conv1(fusion_features)))

        return fusion_features


def Feature_Gather(feature_map, xy):
    """
    :param xy:(B,N,2)  normalize to [-1,1]
    :param feature_map:(B,C,H,W)
    :return:
    """

    # use grid_sample for this.
    # xy(B,N,2)->(B,1,N,2)
    xy = xy.unsqueeze(1)

    interpolate_feature = grid_sample(feature_map, xy)  # (B,C,1,N)

    return interpolate_feature.squeeze(2)  # (B,C,N)


def get_location_img(img, use_location):
    if use_location:
        tmp_img = img
        B, _, H, W = tmp_img.shape
        H_range, W_range = torch.range(0, H - 1).reshape(1, 1, H, 1), \
                           torch.range(0, W - 1).reshape(1, 1, 1, W)
        H_range, W_range = H_range.to(tmp_img), W_range.to(tmp_img)
        H_range = H_range / (H - 1.0) * 2.0 - 1.
        W_range = W_range / (W - 1.0) * 2. - 1.
        H_range, W_range = H_range.expand(B, 1, H, W), W_range.expand(B, 1, H, W)
        location = torch.cat([H_range, W_range], dim=1)
        tmp_img = torch.cat([tmp_img, location], dim=1)
    else:
        tmp_img = img
    return tmp_img


class LocalConv2d(nn.Module):
    def __init__(self, num_rows, num_feats_in, num_feats_out, kernel=1, padding=0, orientation='h'):
        """
        """
        super(LocalConv2d, self).__init__()

        self.num_rows = num_rows
        self.out_channels = num_feats_out
        self.kernel = kernel
        self.pad = padding
        assert orientation in ['w', 'h'], 'orientation should be in ["w","h"].'
        self.h_ori = orientation == 'h'
        self.group_conv = nn.Conv2d(num_feats_in * num_rows, num_feats_out * num_rows, kernel, stride=1,
                                    groups=num_rows)

    def forward(self, x):
        b, c, h, w = x.size()

        if self.pad:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)

        if self.h_ori:
            t = int(h / self.num_rows)

            # unfold by rows
            # (B,C,H/t,W,t)
            x = x.unfold(2, t + self.pad * 2, t)
            # (B,H/t,C,t,W)
            x = x.permute([0, 2, 1, 4, 3]).contiguous()
            x = x.view(b, c * self.num_rows, t + self.pad * 2, (w + self.pad * 2)).contiguous()

            # group convolution for efficient parallel processing
            y = self.group_conv(x)
            y = y.view(b, self.num_rows, self.out_channels, t, w).contiguous()
            y = y.permute([0, 2, 1, 3, 4]).contiguous()
            y = y.view(b, self.out_channels, h, w)
        else:
            t = int(w / self.num_rows)
            # (B,C,H,W/t,t)
            x = x.unfold(3, t + self.pad * 2, t)
            # (B,W/t,C,H,t)
            x = x.permute([0, 3, 1, 2, 4]).contiguous()
            x = x.view(b, c * self.num_rows, h + self.pad * 2, (t + self.pad * 2)).contiguous()

            # group convolution for efficient parallel processing
            y = self.group_conv(x)
            y = y.view(b, self.num_rows, self.out_channels, h, t).contiguous()
            # (B,C,H,W/t,t)
            y = y.permute([0, 2, 3, 1, 4]).contiguous()
            y = y.view(b, self.out_channels, h, w)

        return y


class LocationAwareConv(nn.Module):
    def __init__(self, channel_in, channel_out, split_nums, ori):
        super(LocationAwareConv, self).__init__()
        self.local_conv = LocalConv2d(num_rows=split_nums, num_feats_in=channel_in, num_feats_out=channel_out, kernel=3,
                                      padding=1, orientation=ori)
        self.global_conv = conv3x3(channel_in, channel_out, stride=1)

    def forward(self, img):
        local_out = self.local_conv(img)
        global_out = self.global_conv(img)
        out = local_out + global_out
        return out


class LocationAwareBlock(nn.Module):
    def __init__(self, channel_in, channel_out, split_nums, stride,ori='w'):
        super(LocationAwareBlock, self).__init__()
        self.locationAwareConv = LocationAwareConv(channel_in, channel_out, split_nums,ori)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(channel_out)
        self.conv2 = conv3x3(channel_out, channel_out, stride=stride)

    def forward(self, img):
        out = self.locationAwareConv(img)
        out = self.relu(self.bn(out))
        out = self.conv2(out)
        return out

if __name__ == '__main__':
    block  = LocationAwareBlock(10,20,8,2)
    x= torch.randn((2,10,384,1280)).float()
    y = block(x)
    print(y.shape)
