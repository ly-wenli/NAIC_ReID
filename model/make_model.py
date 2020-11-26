import torch
import torch.nn as nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.resnest import resnest50,resnest50_ibn,resnest101,resnest101_ibn
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a
from .backbones.se_resnet_ibn_a import se_resnet101_ibn_a,se_resnet50_ibn_a
from .backbones.resnet_ibn_b import resnet101_ibn_b,resnet50_ibn_b
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from efficientnet_pytorch import EfficientNet

class GeM(nn.Module):

    def __init__(self, p=3.0, eps=1e-6, freeze_p=True):
        super(GeM, self).__init__()
        self.p = p if freeze_p else Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (1, 1)).pow(1. / self.p)

    def __repr__(self):
        if isinstance(self.p, float):
            p = self.p
        else:
            p = self.p.data.tolist()[0]
        return self.__class__.__name__ +\
               '(' + 'p=' + '{:.4f}'.format(p) +\
               ', ' + 'eps=' + str(self.eps) + ')'

class ClassBlock(nn.Module):
    def __init__(self,input_dim,num_features=512,relu=True):
        super(ClassBlock,self).__init__()
        add_block = []
        add_block += [nn.Conv2d(input_dim,num_features,kernel_size=1,bias=False)]
        add_block += [nn.BatchNorm2d(num_features)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.add_block = add_block
    def forward(self,x):
        x = self.add_block(x)
        x = torch.squeeze(x)
        return x
class PCB(nn.Module):
    def __init__(self,cfg,num_features,num_classes,dropout,out_planes,cut_at_pooling=False):
        super(PCB,self).__init__()
        self.cfg = cfg
        self.num_features = num_features
        self.num_classes = num_classes
        self.dropout = dropout
        self.cut_at_pooling = cut_at_pooling
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.local_conv = nn.Conv2d(out_planes,self.num_features,kernel_size=1,padding=0,bias=False)
        self.local_conv_list = nn.ModuleList()
        for i in range(6):
            self.local_conv_list.append(ClassBlock(out_planes,self.num_features))
        nn.init.kaiming_normal_(self.local_conv.weight,mode='fan_out')
        self.feat_bn2d = nn.BatchNorm2d(self.num_features)
        nn.init.constant_(self.feat_bn2d.weight,1)
        nn.init.constant_(self.feat_bn2d.bias,0)

        self.dy_weight0 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(out_planes,self.num_features,kernel_size=1),
            nn.Conv2d(self.num_features,self.num_features//16,kernel_size=1),
            nn.Conv2d(self.num_features//16,self.num_features,kernel_size=1),
            nn.Flatten(),
            nn.Linear(self.num_features,1)
        )
        self.dy_weight1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_planes, self.num_features, kernel_size=1),
            nn.Conv2d(self.num_features, self.num_features // 16, kernel_size=1),
            nn.Conv2d(self.num_features // 16, self.num_features, kernel_size=1),
            nn.Flatten(),
            nn.Linear(self.num_features, 1)
        )
        self.dy_weight2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_planes, self.num_features, kernel_size=1),
            nn.Conv2d(self.num_features, self.num_features // 16, kernel_size=1),
            nn.Conv2d(self.num_features // 16, self.num_features, kernel_size=1),
            nn.Flatten(),
            nn.Linear(self.num_features, 1)
        )
        self.dy_weight3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_planes, self.num_features, kernel_size=1),
            nn.Conv2d(self.num_features, self.num_features // 16, kernel_size=1),
            nn.Conv2d(self.num_features // 16, self.num_features, kernel_size=1),
            nn.Flatten(),
            nn.Linear(self.num_features, 1)
        )
        self.dy_weight4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_planes, self.num_features, kernel_size=1),
            nn.Conv2d(self.num_features, self.num_features // 16, kernel_size=1),
            nn.Conv2d(self.num_features // 16, self.num_features, kernel_size=1),
            nn.Flatten(),
            nn.Linear(self.num_features, 1)
        )
        self.dy_weight5 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_planes, self.num_features, kernel_size=1),
            nn.Conv2d(self.num_features, self.num_features // 16, kernel_size=1),
            nn.Conv2d(self.num_features // 16, self.num_features, kernel_size=1),
            nn.Flatten(),
            nn.Linear(self.num_features, 1)
        )

        ##-----------------------------stipe1-begin-------------------------------#
        self.instance0 = nn.Linear(self.num_features,self.num_classes)
        nn.init.normal_(self.instance0.weight,std=0.001)
        nn.init.constant_(self.instance0.bias,0)
        ##-----------------------------stipe1-end-------------  ------------------#

        ##-----------------------------stipe1-begin-------------------------------#
        self.instance1 = nn.Linear(self.num_features, self.num_classes)
        nn.init.normal_(self.instance1.weight, std=0.001)
        nn.init.constant_(self.instance1.bias, 0)
        ##-----------------------------stipe1-end-------------  ------------------#

        ##-----------------------------stipe1-begin-------------------------------#
        self.instance2 = nn.Linear(self.num_features, self.num_classes)
        nn.init.normal_(self.instance2.weight, std=0.001)
        nn.init.constant_(self.instance2.bias, 0)
        ##-----------------------------stipe1-end-------------  ------------------#

        ##-----------------------------stipe1-begin-------------------------------#
        self.instance3 = nn.Linear(self.num_features, self.num_classes)
        nn.init.normal_(self.instance3.weight, std=0.001)
        nn.init.constant_(self.instance3.bias, 0)
        ##-----------------------------stipe1-end-------------  ------------------#

        ##-----------------------------stipe1-begin-------------------------------#
        self.instance4 = nn.Linear(self.num_features, self.num_classes)
        nn.init.normal_(self.instance4.weight, std=0.001)
        nn.init.constant_(self.instance4.bias, 0)
        ##-----------------------------stipe1-end-------------  ------------------#

        ##-----------------------------stipe1-begin-------------------------------#
        self.instance5 = nn.Linear(self.num_features, self.num_classes)
        nn.init.normal_(self.instance5.weight, std=0.001)
        nn.init.constant_(self.instance5.bias, 0)
        ##-----------------------------stipe1-end-------------  ------------------#

        ##-----------------------------stipe1-begin-------------------------------#
        self.instance_merge = nn.Linear(self.num_features*6, self.num_classes)
        nn.init.normal_(self.instance_merge.weight, std=0.001)
        nn.init.constant_(self.instance_merge.bias, 0)
        ##-----------------------------stipe1-end-------------  ------------------#
        self.drop = nn.Dropout(self.dropout)
        self.adavgpool = nn.AdaptiveAvgPool2d((6,1))
    def forward(self,x):
        # x = self.relu(x)
        # sx = int(x.size(2)/6)
        # kx = int(x.size(2) - sx*5)
        # x = F.avg_pool2d(x,kernel_size=(kx,x.size(3)),stride=(sx,x.size(3)))
        #x = self.bn(x)
        # x = self.adavgpool(x)
        # part = {}
        # part_feat = {}
        # for i in range(6):
        #     part[i] = torch.unsqueeze(x[:,:,i,:],3)
        #     part_feat[i] = self.local_conv_list[i](part[i])
        #
        # d_weight = {}
        # d_weight[0] = self.dy_weight0(x)
        # d_weight[1] = self.dy_weight1(x)
        # d_weight[2] = self.dy_weight2(x)
        # d_weight[3] = self.dy_weight3(x)
        # d_weight[4] = self.dy_weight4(x)
        # d_weight[5] = self.dy_weight5(x)
        # weight_part_feat = {}
        # for i in range(6):
        #     weight_part_feat[i] = part_feat[i] * d_weight[i]
        # c0 = self.instance0(self.relu(part_feat[0]))
        # c1 = self.instance0(self.relu(part_feat[1]))
        # c2 = self.instance0(self.relu(part_feat[2]))
        # c3 = self.instance0(self.relu(part_feat[3]))
        # c4 = self.instance0(self.relu(part_feat[4]))
        # c5 = self.instance0(self.relu(part_feat[5]))




        # is model is test ,use this feature
        # print("*"*100)
        # print(x.shape)
        # print(x)
        # print("*" * 100)
        if not self.cfg.TEST.PCB_GLOBAL_FEAT_ENSEMBLE:
            """
            x = self.drop(x)
            x = self.local_conv(x)
            x = self.feat_bn2d(x)

            out_t = x

            d_weight0 = self.dy_weight0(x)
            d_weight1 = self.dy_weight1(x)
            d_weight2 = self.dy_weight2(x)
            d_weight3 = self.dy_weight3(x)
            d_weight4 = self.dy_weight4(x)
            d_weight5 = self.dy_weight5(x)
            # print("six weight")
            # print(d_weight0.shape, d_weight1.shape, d_weight2.shape, d_weight3.shape, d_weight4.shape, d_weight5.shape)
            test_x = x.chunk(6, 2)
            test_x0 = test_x[0].contiguous().view(test_x[0].size(0), -1)*d_weight0
            test_x1 = test_x[1].contiguous().view(test_x[1].size(0), -1)*d_weight1
            test_x2 = test_x[2].contiguous().view(test_x[2].size(0), -1)*d_weight2
            test_x3 = test_x[3].contiguous().view(test_x[3].size(0), -1)*d_weight3
            test_x4 = test_x[4].contiguous().view(test_x[4].size(0), -1)*d_weight4
            test_x5 = test_x[5].contiguous().view(test_x[5].size(0), -1)*d_weight5
            x = F.relu(x)

            x = x.chunk(6, 2)
            x0 = x[0].contiguous().view(x[0].size(0), -1)
            x1 = x[1].contiguous().view(x[1].size(0), -1)
            x2 = x[2].contiguous().view(x[2].size(0), -1)
            x3 = x[3].contiguous().view(x[3].size(0), -1)
            x4 = x[4].contiguous().view(x[4].size(0), -1)
            x5 = x[5].contiguous().view(x[5].size(0), -1)
            # print(x0.shape)
            weight_x0 = x0*d_weight0
            weight_x1 = x1*d_weight1
            weight_x2 = x2*d_weight2
            weight_x3 = x3*d_weight3
            weight_x4 = x4*d_weight4
            weight_x5 = x5*d_weight5



            # linear feat,dont use in test.
            c0 = self.instance0(x0)
            c1 = self.instance1(x1)
            c2 = self.instance2(x2)
            c3 = self.instance3(x3)
            c4 = self.instance4(x4)
            c5 = self.instance5(x5)
            """
            x = self.adavgpool(x)
            part = {}
            part_feat = {}
            for i in range(6):
                part[i] = torch.unsqueeze(x[:, :, i, :], 3)
                part_feat[i] = self.local_conv_list[i](part[i])

            d_weight = {}
            d_weight[0] = self.dy_weight0(x)
            d_weight[1] = self.dy_weight1(x)
            d_weight[2] = self.dy_weight2(x)
            d_weight[3] = self.dy_weight3(x)
            d_weight[4] = self.dy_weight4(x)
            d_weight[5] = self.dy_weight5(x)
            weight_part_feat = {}
            for i in range(6):
                weight_part_feat[i] = part_feat[i] * d_weight[i]
            c0 = self.instance0(part_feat[0])
            c1 = self.instance1(part_feat[1])
            c2 = self.instance2(part_feat[2])
            c3 = self.instance3(part_feat[3])
            c4 = self.instance4(part_feat[4])
            c5 = self.instance5(part_feat[5])

            pcb_merge_feat = torch.cat([weight_part_feat[0],weight_part_feat[1],weight_part_feat[2],weight_part_feat[3],weight_part_feat[4],weight_part_feat[5]],dim=1)
            # pcb_merge_feat = x0 + x1 + x2 + x3 + x4 + x5
            pcb_merge_feat_train = self.instance_merge(pcb_merge_feat)
            if not self.training:
                if self.cfg.TEST.USE_PCB_MERGE_FEAT:
                    return pcb_merge_feat
                else:
                    return (weight_part_feat[0],weight_part_feat[1],weight_part_feat[2],weight_part_feat[3],weight_part_feat[4],weight_part_feat[5])
                # val_c0 = c0.view(c0.shape[0], -1)
                # val_c1 = c1.view(c1.shape[0], -1)
                # val_c2 = c2.view(c2.shape[0], -1)
                # val_c3 = c3.view(c3.shape[0], -1)
                # val_c4 = c4.view(c4.shape[0], -1)
                # val_c5 = c5.view(c5.shape[0], -1)
                # return (val_c0, val_c1, val_c2, val_c3, val_c4, val_c5)
            if self.cfg.MODEL.MERGE_PCB_FEAT:

                return (c0, c1, c2, c3, c4, c5, pcb_merge_feat_train)
            else:
                return (c0, c1, c2, c3, c4, c5)
        else:
            # this if modoule is use pcb global feature to ensemble distmat.
            if not self.training:
                # print("x.shape",x.shape)

                # use local_conv can raise 0.1 sorce:0.38(dont use is 0.37)

                x = self.local_conv(x)
                out0 = x / x.norm(2, 1).unsqueeze(1).expand_as(x)

                # wenli:use this GeM pooling can raise 0.2(use local_conv) sorce:0.37(not use is 0.36)

                out0 = GeM()(out0)
                out0 = out0.view(out0.shape[0], -1)

                # wenli:the next score is 0.07,dont use next method
                # x = self.local_conv(x)
                # x = self.feat_bn2d(x)
                # out0 = F.relu(x)
                # out0 = out0.view(out0.shape[0], -1)

                return out0
            else:
                x = self.drop(x)
                x = self.local_conv(x)
                x = self.feat_bn2d(x)

                d_weight0 = self.dy_weight(x)
                d_weight1 = self.dy_weight(x)
                d_weight2 = self.dy_weight(x)
                d_weight3 = self.dy_weight(x)
                d_weight4 = self.dy_weight(x)
                d_weight5 = self.dy_weight(x)

                x = F.relu(x)

                x = x.chunk(6, 2)
                x0 = x[0].contiguous().view(x[0].size(0), -1)
                x1 = x[1].contiguous().view(x[1].size(0), -1)
                x2 = x[2].contiguous().view(x[2].size(0), -1)
                x3 = x[3].contiguous().view(x[3].size(0), -1)
                x4 = x[4].contiguous().view(x[4].size(0), -1)
                x5 = x[5].contiguous().view(x[5].size(0), -1)


                weight_x0 = x0 * d_weight0
                weight_x1 = x1 * d_weight1
                weight_x2 = x2 * d_weight2
                weight_x3 = x3 * d_weight3
                weight_x4 = x4 * d_weight4
                weight_x5 = x5 * d_weight5


                c0 = self.instance0(x0)*0.1
                c1 = self.instance1(x1)*0.2
                c2 = self.instance2(x2)*0.5
                c3 = self.instance3(x3)*0.33
                c4 = self.instance4(x4)*0.8
                c5 = self.instance5(x5)*0.22

                pcb_merge_feat = torch.cat([weight_x0, weight_x1, weight_x2, weight_x3, weight_x4, weight_x5], dim=1)
                # pcb_merge_feat = x0 + x1 + x2 + x3 + x4 + x5
                pcb_merge_feat_train = self.instance_merge(pcb_merge_feat)
                if self.cfg.MODEL.MERGE_PCB_FEAT:

                    return (c0, c1, c2, c3, c4, c5, pcb_merge_feat_train)
                else:
                    return (c0, c1, c2, c3, c4, c5)






def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        self.cfg = cfg
        self.model_name = model_name
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        # self.in_planes = 1280
        # model_weight_b0 = EfficientNet.from_pretrained('efficientnet-b0')
        # model_weight_b0.to('cuda')
        # mm = nn.Sequential(*model_weight_b0.named_children())
        # self.base = model_weight_b0.extract_features
        #
        # from IPython import embed
        # embed()
        #print('using efficientnet-b0 as a backbone')

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, frozen_stages=cfg.MODEL.FROZEN,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet101_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet101_ibn_a(last_stride)
            print('using se_resnet101_ibn_a as a backbone')
        elif model_name == 'se_resnet50_ibn_a':
            self.in_planes = 2048
            self.base = se_resnet50_ibn_a(last_stride)
            print('using se_resnet101_ibn_a as a backbone')
        elif model_name == 'resnet101_ibn_b':
            self.in_planes = 2048
            self.base = resnet101_ibn_b(last_stride)
            print('using resnet101_ibn_b as a backbone')
        elif model_name == 'resnet50_ibn_b':
            self.in_planes = 2048
            self.base = resnet50_ibn_b(last_stride)
            print('using resnet50_ibn_b as a backbone')
        elif model_name == 'resnest50':
            self.in_planes = 2048
            self.base = resnest50(last_stride)
            print('using resnest50 as a backbone')
        elif model_name == 'resnest50_ibn':
            self.in_planes = 2048
            self.base = resnest50_ibn(last_stride)
            print('using resnest50_ibn as a backbone')
        elif model_name == 'resnest101':
            self.in_planes = 2048
            self.base = resnest101(last_stride)
            print('using resnest101 as a backbone')
        elif model_name == 'resnest101_ibn':
            self.in_planes = 2048
            self.base = resnest101_ibn(last_stride)
            print('using resnest101_ibn as a backbone')
        elif model_name == 'efficientnet_b7':
            # self.in_planes = 1280
            #
            # model_weight_b0 = EfficientNet.from_pretrained('efficientnet-b0')
            # model_weight_b0.to('cuda')
            # self.base = model_weight_b0.extract_features
            self.base = EfficientNet.from_pretrained('efficientnet-b0')
            self.in_planes = self.base._fc.in_features
            print('using efficientnet-b0 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet' and model_name != 'efficientnet_b7':
            # if model_name == 'efficientnet_b7':
            #     state_dict = torch.load(model_path)
            #     # self.base.load_state_dict(state_dict)
            #     if 'state_dict' in state_dict:
            #         param_dict = state_dict['state_dict']
            #     for i in param_dict:
            #         if 'fc' in i:
            #             continue
            #         self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            # else:
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        if cfg.MODEL.POOLING_METHOD == 'GeM':
            print('using GeM pooling')
            self.gap = GeM()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
        if cfg.MODEL.IF_USE_PCB:
            self.pcb = PCB(cfg,256,num_classes,0.5,self.in_planes,cut_at_pooling=False)
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        # from IPython import embed
        # embed()
    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        # device = 'cuda'
        # x.to(device)
        # from IPython import  embed
        # embed()
        #print("x.shape",x.shape)
        if 'efficientnet_b7' == self.model_name:
            x = self.base.extract_features(x)
        else:

            x = self.base(x)
        if self.cfg.MODEL.IF_USE_PCB:
            pcb_out = self.pcb(x)

        # print("x.shape",x.shape)
        global_feat = self.gap(x)
        # print("global_feat.shape",global_feat.shape)
        # print("pcb_out.shape",pcb_out.shape)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
            if self.cfg.MODEL.IF_USE_PCB:
                return cls_score, global_feat, pcb_out
            else:
                return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                if self.cfg.MODEL.IF_USE_PCB:
                    return feat, pcb_out
                else:
                    return feat
            else:
                # print("Test with feature before BN")
                if self.cfg.MODEL.IF_USE_PCB:
                    return global_feat, pcb_out
                else:
                    return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i.replace('module.','')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
