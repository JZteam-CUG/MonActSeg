import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from models.net_utils.tgcn import ConvTemporalGraphical
from models.net_utils.graph import Graph
from models.fusion_modules import Fusion_Attention_Gate

class Ske_Pre_Model(nn.Module):
    def __init__(
        self,
        in_channels=4,
        num_class=8,
        dil=[1, 2, 4, 8, 16],
        filters=64,
        edge_importance_weighting=True,
        **kwargs,
    ):
        super(Ske_Pre_Model, self).__init__()
        graph_args = {'layout': 'monkey', 'strategy': 'spatial'}
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 3
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.conv_1x1 = nn.Conv2d(in_channels, filters, 1)
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[0], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[1], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[2], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[3], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[4], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[5], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[6], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[7], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[8], residual=True),
            st_gcn(filters, filters, kernel_size, 1, A=A, dilation=dil[9], residual=True),
        ))
        self.dropout = nn.Dropout(p=0.2)

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.conv_out = nn.Conv1d(filters, num_class, kernel_size=1)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        x = self.conv_1x1(x)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
            x = self.dropout(x)

        x = F.avg_pool2d(x, kernel_size=(1, V))

        c = x.size(1)
        t = x.size(2)
        x = x.view(N, M, c, t).mean(dim=1).view(N, c, t)
        out = self.conv_out(x)
        return out, x


class st_gcn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        A=None,
        dilation=1,
        residual=True,
    ):
        super(st_gcn, self).__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        pad = int((dilation * (kernel_size[0] - 1)) / 2)
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(kernel_size[0], 1),
                stride=(stride, 1),
                padding=(pad, 0),
                dilation=(dilation, 1),
            ),
            nn.BatchNorm2d(out_channels),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x)
        x = self.relu(x)
        x = x + res
        return x, A

class RGB_Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(RGB_Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out,f

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        output = self.conv_out(out) * mask[:, 0:1, :]
        return output,out
    
class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x,mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)*mask[:, 0:1, :]

class MultiStageModel(nn.Module):
    def __init__(
        self,
        dil,
        num_layers_RF,
        num_R,
        num_f_maps,
        skeleton_features_dim,
        rgb_features_dim,
        num_classes,
        num_layers_PG,
        feedback_features=False,
    ):
        super(MultiStageModel, self).__init__()

        self.skeleton_PG = Ske_Pre_Model(
            in_channels=skeleton_features_dim,
            num_class=num_classes,
            filters=num_f_maps,
            dil=dil,
        )
        self.rgb_PG = RGB_Prediction_Generation(
            num_layers=num_layers_PG,
            num_f_maps=num_f_maps,
            dim=rgb_features_dim,
            num_classes=num_classes,
        )

        print("Initializing MultiStageModel with Fusion_Attention_Gate")
        self.use_feature_feedback = feedback_features

        refine_input_dim = num_f_maps if self.use_feature_feedback else num_classes
        self.skeleton_Rs = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_RF, num_f_maps, refine_input_dim, num_classes)) for _ in range(num_R)]
        )
        self.rgb_Rs = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_RF, num_f_maps, refine_input_dim, num_classes)) for _ in range(num_R)]
        )
        self.fusion_modules = nn.ModuleList()

        for _ in range(num_R + 1):
            self.fusion_modules.append(
                Fusion_Attention_Gate(num_f_maps, num_f_maps, num_f_maps, num_classes)
            )

    def _run_fusion(self, module, ske_feat, rgb_feat, mask):
        if self.use_feature_feedback:
            fused_out, feedback_s, feedback_r = module(
                ske_feat, rgb_feat, mask=mask, return_features=True
            )
            return fused_out, feedback_s, feedback_r
        fused_out = module(ske_feat, rgb_feat, mask=mask)
        return fused_out, None, None

    def forward(self, skeleton_input, rgb_input, mask):
        skeleton_outputs = []
        rgb_outputs = []
        router_weights_list = []
        weighted_features_list = []
        feedback_s = None
        feedback_r = None

        skeleton_out, skeleton_feature = self.skeleton_PG(skeleton_input)
        rgb_out, rgb_feature = self.rgb_PG(rgb_input)

        skeleton_outputs.append(skeleton_out)
        rgb_outputs.append(rgb_out)

        fused_out, feedback_s, feedback_r = self._run_fusion(
            self.fusion_modules[0], skeleton_feature, rgb_feature, mask
        )
        fused_outputs = fused_out.unsqueeze(0)

        for i, (skeleton_R, rgb_R) in enumerate(zip(self.skeleton_Rs, self.rgb_Rs)):
            if self.use_feature_feedback:
                ske_in_next = feedback_s
                rgb_in_next = feedback_r
            else:
                ske_in_next = F.softmax(skeleton_out, dim=1)
                rgb_in_next = F.softmax(rgb_out, dim=1)

            skeleton_out, skeleton_feature = skeleton_R(ske_in_next, mask)
            rgb_out, rgb_feature = rgb_R(rgb_in_next, mask)

            skeleton_outputs.append(skeleton_out)
            rgb_outputs.append(rgb_out)

            fused_out, feedback_s, feedback_r = self._run_fusion(
                self.fusion_modules[i + 1], skeleton_feature, rgb_feature, mask
            )
            fused_outputs = torch.cat((fused_outputs, fused_out.unsqueeze(0)), dim=0)

        return fused_outputs, skeleton_outputs, rgb_outputs, router_weights_list, weighted_features_list


