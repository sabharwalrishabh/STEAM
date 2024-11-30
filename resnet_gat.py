import torch.nn as nn
from torchvision.models import ResNet
from gnn import ChannelGAT, SpatialGAT
from new_utils import create_edge_list, global_avg_pooling_to_7x7, SpecialConv,\
SingleKernelConv, conv1x1, create_spatial_adjacency_matrix, drop_random_edges, create_directed_edge_list_5
from torch_geometric.data import Data, Batch
from resnet_cbam import SAM
import torch.nn.functional as F
import torch
from gnnmha import GNNMHA


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class CA_GAT(nn.Module):
    def __init__(self, in_channel, hidden_dim, in_ch_for_conv, args, kernel_size=7):
        super(CA_GAT, self).__init__()
        # self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size, padding=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # self.ca_gat = ChannelGAT(1, hidden_dim)
        self.ca_gat = GNNMHA(1,8)
        # self.conv = SingleKernelConv(in_ch_for_conv)
        self.edge_indices = create_edge_list(0, in_ch_for_conv).cuda(args.gpu)  # no need of BS
        # self.edge_indices = create_directed_edge_list_5(in_ch_for_conv).cuda(args.gpu)
        # print("3 neighbor CA | avg-max concat for CA")

    def forward(self, input_feat):
        b, c, _, _ = input_feat.size()
        feature = self.avg_pool(input_feat).reshape(b, c, -1)
        # feature_avg = self.avg_pool(input_feat).reshape(b, c, -1)
        # feature_max = self.max_pool(input_feat).reshape(b, c, -1)
        # feature = torch.cat([feature_avg, feature_max], dim=-1)
        # edge_indices = create_edge_list(b, c)
        data_list = [Data(x=feature[i], edge_index=self.edge_indices) for i in range(b)]
        batched_graph = Batch.from_data_list(data_list)
        batched_scores = self.ca_gat(batched_graph).view(b, c, 1, 1)
        updated_cnn_output = input_feat * batched_scores.expand_as(input_feat)
        # return cnn_output + updated_cnn_output
        return updated_cnn_output


class SA_GAT(nn.Module):
    def __init__(self, args, output_kernel_size=7):
        super(SA_GAT, self).__init__()
        self.size = output_kernel_size
        # self.sa_gat = SpatialGAT(1, 1)
        # self.sa_gat = SpatialGAT(1, 1)
        self.sa_gat = GNNMHA(1,8)
        self.spatial_edge_indices = create_spatial_adjacency_matrix(self.size)
        #self.conv = conv1x1()
        # self.conv = SpecialConv()
        self.args = args

    def forward(self, input_feat):
        b, c, h, w = input_feat.size()
        feature = global_avg_pooling_to_7x7(input_feat).reshape(b, -1, 1)
        # feature = self.conv(global_avg_pooling_to_7x7(input_feat).reshape(b, 2, 7, 7)).reshape(b, -1, 1)
        edge_indices = self.spatial_edge_indices
        if self.training:
            edge_indices = drop_random_edges(self.spatial_edge_indices)
            # edge_indices = edge_indices.cuda(self.args.gpu)
        edge_indices = edge_indices.cuda(self.args.gpu)
        # spatial_edge_indices = create_spatial_adjacency_matrix_batch(b, self.size)
        data_list = [Data(x=feature[i], edge_index=edge_indices) for i in range(b)]
        batched_graph = Batch.from_data_list(data_list)
        batched_scores = self.sa_gat(batched_graph).view(b, 1, self.size, self.size)

        batched_scores = batched_scores.repeat_interleave(h//self.size, dim=2).repeat_interleave(w//self.size, dim=3)
        updated_cnn_output = input_feat * batched_scores.expand_as(input_feat)
        return updated_cnn_output


class GATLayer(nn.Module):
    def __init__(self, in_channel, hidden_dim, in_ch_for_conv, args, kernel_size=7, order="ca-sa"):
        super(GATLayer, self).__init__()
        """ 
        order decides position of CA and SA module
        'ca-sa', 'sa-ca' or 'ca+sa'   
        """
        self.order = order
        print(self.order)
        self.channel_gat = CA_GAT(in_channel, hidden_dim, in_ch_for_conv, args, kernel_size=kernel_size)
        self.spatial_gat = SA_GAT(args)
        # self.conv1x1 = nn.Sequential(SingleKernelConv(in_ch_for_conv),
        #                              nn.BatchNorm2d(in_ch_for_conv),
        #                              nn.ReLU(inplace=True))


    def forward(self, cnn_output):
        if self.order == "ca-sa":
            # channel_adjusted_fea = self.conv1x1(self.channel_gat(cnn_output))
            channel_adjusted_fea = F.tanh(self.channel_gat(cnn_output))
            spatial_adjusted_fea = self.spatial_gat(channel_adjusted_fea)
            return cnn_output + spatial_adjusted_fea
        elif self.order == "sa-ca":
            spatial_adjusted_fea = F.tanh(self.spatial_gat(cnn_output)) 
            channel_adjusted_fea = self.channel_gat(spatial_adjusted_fea)
            return cnn_output + channel_adjusted_fea
        elif self.order == "ca+sa":
            spatial_adjusted_fea = self.spatial_gat(cnn_output)
            channel_adjusted_fea = self.channel_gat(cnn_output)
            gate_value = self.gate(spatial_adjusted_fea + channel_adjusted_fea)
            return gate_value * channel_adjusted_fea + (1 - gate_value) * spatial_adjusted_fea + cnn_output
        else:
            raise NotImplementedError(f"order {self.order} is incorrect, only ['ca-sa', 'sa-ca' or 'ca+sa'] supported")


class GATBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, gnn_in_ch = True, args=None):
        super(GATBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        if gnn_in_ch:
            self.gat_module = GATLayer(1, 1, planes, args)
        else:
            self.gat_module = None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.gat_module is not None:
            out = self.gat_module(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GATBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *,gnn_in_ch = True, args=None):
        super(GATBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        if gnn_in_ch:
            self.gat_module = GATLayer(1, 1, planes*4, args)
        else:
            self.gat_module = None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.gat_module is not None:
            out = self.gat_module(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, args=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], args)
        self.layer2 = self._make_layer(block, 128, layers[1], args, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], args, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], args, stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck) and m.bn3.weight is not None:
        #             nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
        #         elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #             nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        #----ECA AUTHORS DID THIS-------#
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, args, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, gnn_in_ch = False, args=args))
        self.inplanes = planes * block.expansion
        if blocks == 6:
            insert_at = [2, 5]
        else:
            insert_at = [blocks-1]

        for i in range(1, blocks):
            if i in insert_at:
                layers.append(block(self.inplanes, planes, gnn_in_ch = True, args=args))
            else:
                layers.append(block(self.inplanes, planes, gnn_in_ch = False, args=args))

        # if planes!=512:
        #     layers.append(block(self.inplanes, planes, stride, downsample, gnn_in_ch = False, args=args))
        #     self.inplanes = planes * block.expansion
        #     for i in range(1, blocks):
        #         layers.append(block(self.inplanes, planes, gnn_in_ch = False, args=args))
        # else:
        #     layers.append(block(self.inplanes, planes, stride, downsample, gnn_in_ch = True, args=args))
        #     self.inplanes = planes * block.expansion
        #     for i in range(1, blocks):
        #         layers.append(block(self.inplanes, planes, gnn_in_ch = True, args=args))
                
        
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def gat_resnet18(num_classes=1000, args=None):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(GATBasicBlock, [2, 2, 2, 2], num_classes=num_classes, args=args)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def gat_resnet34(num_classes=1000, args=None):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(GATBasicBlock, [3, 4, 6, 3], num_classes=num_classes,args=args)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def gat_resnet50(num_classes=1000, args=None):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    model = ResNet(GATBottleneck, [3, 4, 6, 3], num_classes=num_classes, args=args)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def gat_resnet101(num_classes=1000, args=None):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(GATBottleneck, [3, 4, 23, 3], num_classes=num_classes, args=args)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def gat_resnet152(num_classes=1000, args=None):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(GATBottleneck, [3, 8, 36, 3], num_classes=num_classes, args=args)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
