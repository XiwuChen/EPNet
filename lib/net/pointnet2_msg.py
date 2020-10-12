import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_lib.pointnet2.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from lib.config import cfg
from lib.net.ConvBlocks import Feature_Gather,Fusion_Conv,Atten_Fusion_Conv,BasicBlock,get_location_img,LocationAwareBlock






class Pointnet2MSG(nn.Module):
    def __init__(self, input_channels=6, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels
        self.size_range = [1280.0, 384.0]  # if cfg.LI_FUSION.IMG_SHIFT==False else [1280.0,

        skip_channel_list = [input_channels]
        for k in range(cfg.RPN.SA_CONFIG.NPOINTS.__len__()):
            mlps = cfg.RPN.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=cfg.RPN.SA_CONFIG.NPOINTS[k],
                    radii=cfg.RPN.SA_CONFIG.RADIUS[k],
                    nsamples=cfg.RPN.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=use_xyz,
                    bn=cfg.RPN.USE_BN
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        ##################
        if cfg.LI_FUSION.ENABLED:
            self.use_location = cfg.LI_FUSION.IMG_LOCATION
            if self.use_location:
                extra_img_channel = 2
            else:
                extra_img_channel = 0

            self.Img_Block = nn.ModuleList()
            self.Fusion_Conv = nn.ModuleList()
            self.DeConv = nn.ModuleList()
            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                # if cfg.LI_FUSION.CONV7 and i == 0:
                #     self.Img_Block.append(
                #         Conv7x7Block(cfg.LI_FUSION.IMG_CHANNELS[i] + extra_img_channel,
                #                      cfg.LI_FUSION.IMG_CHANNELS[i + 1],
                #                      stride=2,use_relu=cfg.LI_FUSION.DOUBLE_RELU))
                # else:
                if cfg.LI_FUSION.LOCAL_CONV:
                    self.Img_Block.append(
                        LocationAwareBlock(cfg.LI_FUSION.IMG_CHANNELS[i] + extra_img_channel, cfg.LI_FUSION.IMG_CHANNELS[i + 1],split_nums=8,stride=2,ori=cfg.LI_FUSION.ORI))
                else:
                    self.Img_Block.append(
                        BasicBlock(cfg.LI_FUSION.IMG_CHANNELS[i] + extra_img_channel, cfg.LI_FUSION.IMG_CHANNELS[i + 1],
                                   stride=1, use_relu=cfg.LI_FUSION.DOUBLE_RELU, layer_nums=cfg.LI_FUSION.LAYER_NUM))
                if cfg.LI_FUSION.ADD_Image_Attention:
                    self.Fusion_Conv.append(
                        Atten_Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1], cfg.LI_FUSION.POINT_CHANNELS[i],
                                          cfg.LI_FUSION.POINT_CHANNELS[i]))
                else:
                    self.Fusion_Conv.append(
                        Fusion_Conv(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + cfg.LI_FUSION.POINT_CHANNELS[i],
                                    cfg.LI_FUSION.POINT_CHANNELS[i]))

                self.DeConv.append(nn.ConvTranspose2d(cfg.LI_FUSION.IMG_CHANNELS[i + 1] + extra_img_channel,
                                                      cfg.LI_FUSION.DeConv_Reduce[i],
                                                      kernel_size=cfg.LI_FUSION.DeConv_Kernels[i],
                                                      stride=cfg.LI_FUSION.DeConv_Kernels[i]))

            self.image_fusion_conv = nn.Conv2d(sum(cfg.LI_FUSION.DeConv_Reduce),
                                               cfg.LI_FUSION.IMG_FEATURES_CHANNEL // 4, kernel_size=1)
            self.image_fusion_bn = torch.nn.BatchNorm2d(cfg.LI_FUSION.IMG_FEATURES_CHANNEL // 4)

            if cfg.LI_FUSION.ADD_Image_Attention:
                self.final_fusion_img_point = Atten_Fusion_Conv(cfg.LI_FUSION.IMG_FEATURES_CHANNEL // 4,
                                                                cfg.LI_FUSION.IMG_FEATURES_CHANNEL,
                                                                cfg.LI_FUSION.IMG_FEATURES_CHANNEL)
            else:
                self.final_fusion_img_point = Fusion_Conv(
                    cfg.LI_FUSION.IMG_FEATURES_CHANNEL + cfg.LI_FUSION.IMG_FEATURES_CHANNEL // 4,
                    cfg.LI_FUSION.IMG_FEATURES_CHANNEL)

        self.FP_modules = nn.ModuleList()

        for k in range(cfg.RPN.FP_MLPS.__len__()):
            pre_channel = cfg.RPN.FP_MLPS[k + 1][-1] if k + 1 < len(cfg.RPN.FP_MLPS) else channel_out
            self.FP_modules.append(
                PointnetFPModule(mlp=[pre_channel + skip_channel_list[k]] + cfg.RPN.FP_MLPS[k])
            )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, image=None, xy=None):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]

        if cfg.LI_FUSION.ENABLED:
            #### normalize xy to [-1,1]

            xy[:, :, 0] = xy[:, :, 0] / (self.size_range[0] - 1.0) * 2.0 - 1.0
            xy[:, :, 1] = xy[:, :, 1] / (self.size_range[1] - 1.0) * 2.0 - 1.0  # = xy / (size_range - 1.) * 2 - 1.
            l_xy_cor = [xy]
            img = [image]

        for i in range(len(self.SA_modules)):
            li_xyz, li_features, li_index = self.SA_modules[i](l_xyz[i], l_features[i])

            if cfg.LI_FUSION.ENABLED:
                li_index = li_index.long().unsqueeze(-1).repeat(1, 1, 2)
                li_xy_cor = torch.gather(l_xy_cor[i], 1, li_index)

                tmp_img = get_location_img(img[i], self.use_location)
                image = self.Img_Block[i](tmp_img)

                # print(image.shape)
                img_gather_feature = Feature_Gather(image, li_xy_cor)  # , scale= 2**(i+1))

                li_features = self.Fusion_Conv[i](li_features, img_gather_feature)
                l_xy_cor.append(li_xy_cor)
                img.append(image)

            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        if cfg.LI_FUSION.ENABLED:
            # for i in range(1,len(img))
            DeConv = []
            for i in range(len(cfg.LI_FUSION.IMG_CHANNELS) - 1):
                tmp_img = get_location_img(img[i + 1], self.use_location)
                DeConv.append(self.DeConv[i](tmp_img))
            de_concat = torch.cat(DeConv, dim=1)

            img_fusion = F.relu(self.image_fusion_bn(self.image_fusion_conv(de_concat)))
            img_fusion_gather_feature = Feature_Gather(img_fusion, xy)
            l_features[0] = self.final_fusion_img_point(l_features[0], img_fusion_gather_feature)

        return l_xyz[0], l_features[0]


class Pointnet2MSG_returnMiddleStages(Pointnet2MSG):
    def __init__(self, input_channels=6, use_xyz=True):
        super().__init__(input_channels, use_xyz)

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        idxs = []
        for i in range(len(self.SA_modules)):
            li_xyz, li_features, idx = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)
            idxs.append(idx)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_xyz, l_features, idxs
