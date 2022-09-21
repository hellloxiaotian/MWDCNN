'''
    This model is dynamic conv + wavelet transform + Residual dense block callde DWD
'''

import torch
import torch.nn as nn
from model_common import common
from model_common.WRB import WRB
from model_common.RDB import RDB


def make_model(args):
    return WMDCNN(args), 1

class WMDCNN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(WMDCNN, self).__init__()

        kernel_size = 5
        n_feats = args.n_feats
        dynamic_conv = common.dynamic_conv
        growth_rate = args.growth_rate
        rdb_num_layers = args.RDB_num_layers

        self.conv1 = conv(args.n_colors, n_feats, kernel_size)  # conv1

        self.dy_conv_block = nn.Sequential(
            dynamic_conv(n_feats, n_feats, kernel_size),
            nn.ReLU(True)
        )

        self.conv_block1 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(True)
        )

        self.WRB1 = WRB(args, n_feats)
        self.WRB2 = WRB(args, n_feats)

        self.RDB_1 = nn.Sequential(
            RDB(n_feats, growth_rate, rdb_num_layers),
            nn.ReLU(True)
        )

        self.RDB_2 = nn.Sequential(
            RDB(n_feats, growth_rate, rdb_num_layers),
            nn.ReLU(True)
        )

        self.conv_block2 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),
            nn.ReLU(True)
        )

        self.conv2 = conv(n_feats, args.n_colors, kernel_size)  # conv1

        self.seq = nn.Sequential(
            self.conv1,
            self.dy_conv_block,
            self.conv_block1,
            self.WRB1,
            self.WRB2
        )

    def forward(self, x):
        y = x
        
        out1 = self.seq(x)

        out2 = self.RDB_1(out1)
        out3 = self.RDB_2(out2)

        out4 = out1 + out2 + out3

        out5 = self.conv_block2(out4)
        out = self.conv2(out5)

        return y - out