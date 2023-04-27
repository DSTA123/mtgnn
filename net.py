import torch.nn

from layer import *
import argparse

class gtnet(nn.Module):
    def __init__(self, gcn_true, buildA_true, gcn_depth, num_nodes, device, predefined_A=None, static_feat=None, dropout=0.3,subgraph_size=20, node_dim=40,
                 dilation_exponential=1, in_dim=2, residual_channels=32, conv_channels=32, skip_channels=64, end_channels=128, seq_length=12, out_dim=12,
                 layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True, t_in=19, num_cheb_filter=13, conv_type='cheb', K=3, c_in=32, time_strides=1):
        super(gtnet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.leakyRelu = nn.LeakyReLU(inplace=True)
        self.glu = nn.GLU()
        self.gcn_true = gcn_true
        self.buildA_true = buildA_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.conv_type = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.fcmy1 = nn.ModuleList()
        self.fcmy2 = nn.ModuleList()


        # net_graph = GraphLearn(net_config['num_nodes'], net_config['init_feature_num']).to(device)

        self.gc = graph_constructor(num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat).to(device)

        self.seq_length = seq_length

        self.idx = torch.arange(self.num_nodes).to(device)

        kernel_size = 7
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1  # 19

        for i in range(1):
            if dilation_exponential>1:
                rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                rf_size_i = i*layers*(kernel_size-1)+1  # 1
            new_dilation = 1
            for j in range(1,layers+1):
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)  # 7 13  19

                self.filter_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.gate_convs.append(dilated_inception(residual_channels, conv_channels, dilation_factor=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=residual_channels,
                                                 kernel_size=(1, 1)))
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.seq_length-rf_size_j+1)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, self.receptive_field-rf_size_j+1)))  # (1,13) (1,7) (1,1)

                if self.gcn_true:
                    # self.gconv1.append(GraphConv(conv_channels, residual_channels, conv_type, K=K, activation=None,with_self=False))
                    # self.gconv2.append(GraphConv(conv_channels, residual_channels, conv_type, K=K, activation=None,with_self=False))

                    # self.gconv1.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))
                    # self.gconv2.append(mixprop(conv_channels, residual_channels, gcn_depth, dropout, propalpha))

                    adp = self.gc(self.idx)

                    num_nodes = adp.size(-1)
                    self.temporal_att0 = TemporalAttention(num_nodes, conv_channels, t_in)
                    self.temporal_att1 = TemporalAttention(num_nodes, conv_channels, 13)
                    self.temporal_att2 = TemporalAttention(num_nodes, conv_channels, 7)
                    self.spatial_att0 = SpatialAttention(num_nodes, conv_channels, t_in)
                    self.spatial_att1 = SpatialAttention(num_nodes, conv_channels, 13)
                    self.spatial_att2 = SpatialAttention(num_nodes, conv_channels, 7)
                    self.cheb_conv = ChebConv(32, 32, adp)

                    self.gtu1 = GateDilatedConv(residual_channels, time_strides, 3)
                    self.gtu2 = GateDilatedConv(residual_channels, time_strides, 5)
                    self.gtu3 = GateDilatedConv(residual_channels, time_strides, 7)
                    self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 2), stride=None, padding=0,
                                                      return_indices=False, ceil_mode=False)
                    self.fcmy1.append(nn.Sequential(nn.Linear(3 * (self.receptive_field - rf_size_j + 7) - 12, self.receptive_field - rf_size_j + 7), ))
                    self.fcmy2.append(nn.Sequential(nn.Linear(3 * (self.receptive_field - rf_size_j + 7) - 12, self.receptive_field - rf_size_j + 1), ))
                    # print("out:", 3 * (self.receptive_field - rf_size_j + 7) - 8)

                if self.seq_length>self.receptive_field:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_length - rf_size_j + 1),elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affline))  # (32,207,13)

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        # self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        # if self.gcn_true:
        #     if self.buildA_true:
        #         if idx is None:
        #
        #             adp = self.gc(self.idx)
        #         else:
        #             adp = self.gc(idx)
        #     else:
        #         adp = self.predefined_A

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):

            # print("i:", i)
            # TC Module
            residual = x
            # print("x:", x.size())
            # filter = self.filter_convs[i](x)
            # filter = torch.tanh(filter)
            # # print("filter:", filter.size())
            # gate = self.gate_convs[i](x)
            # gate = torch.sigmoid(gate)
            # # print("gate:", gate.size())
            # x = filter * gate
            # x = F.dropout(x, self.dropout, training=self.training)

            x_gtu = []
            x_gtu.append(self.gtu1(x))  # B,F,N,T
            # print("x_gtu0:", x_gtu[0].size())
            x_gtu.append(self.gtu2(x))  # B,F,N,T-2
            # print("x_gtu1:", x_gtu[1].size())
            x_gtu.append(self.gtu3(x))  # B,F,N,T-6
            # print("x_gtu2:", x_gtu[2].size())
            time_conv = torch.cat(x_gtu, dim=-1)  # B,F,N,3T-8
            # print("time_conv:", time_conv.size())
            time_conv = self.fcmy1[i](time_conv)
            time_conv = F.dropout(time_conv, self.dropout, training=self.training)
            # print("x_gtu:", time_conv.size())
            if x.size(-1) == 1:
                x = self.relu(time_conv)
            else:
                # print("xx:", x.size(-1))
                x = self.relu(x[:, :, :, -time_conv.size(3):] + time_conv)  # B,F,N,T
                # print("x:", x.size())
            #   Module
            s = x
            s = self.skip_convs[i](s)
            # print("s:", s.size())
            # print("skip:", skip.size())
            skip = s + skip
            if self.gcn_true:
                # x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
                x = x.permute(0, 3, 2, 1)
                # print("x:", x.shape)
                b, c, n_d, f = x.size()
                if i == 0:
                    temporal_att = self.temporal_att0(x)
                    x_tat = (torch.matmul(temporal_att, x.reshape(b, c, -1)).reshape(b, c, n_d, f))
                    spatial_att = self.spatial_att0(x_tat)
                    x = self.cheb_conv(x, spatial_att).permute(0, 3, 2, 1)
                if i == 1:
                    temporal_att = self.temporal_att1(x)
                    x_tat = (torch.matmul(temporal_att, x.reshape(b, c, -1)).reshape(b, c, n_d, f))
                    spatial_att = self.spatial_att1(x_tat)
                    x = self.cheb_conv(x, spatial_att).permute(0, 3, 2, 1)
                if i == 2:
                    temporal_att = self.temporal_att2(x)
                    x_tat = (torch.matmul(temporal_att, x.reshape(b, c, -1)).reshape(b, c, n_d, f))
                    spatial_att = self.spatial_att2(x_tat)
                    x = self.cheb_conv(x, spatial_att).permute(0, 3, 2, 1)
            else:
                x = self.residual_convs[i](x)
            # print("x1:", x.size())

            x_gtu = []
            x_gtu.append(self.gtu1(x))  # B,F,N,T
            # print("x_gtu0:", x_gtu[0].size())
            x_gtu.append(self.gtu2(x))  # B,F,N,T-2
            # print("x_gtu1:", x_gtu[1].size())
            x_gtu.append(self.gtu3(x))  # B,F,N,T-6
            # print("x_gtu2:", x_gtu[2].size())
            time_conv = torch.cat(x_gtu, dim=-1)  # B,F,N,3T-8
            # print("time_conv:", time_conv.size())
            time_conv = self.fcmy2[i](time_conv)
            time_conv = F.dropout(time_conv, self.dropout, training=self.training)
            # print("x_gtu:", time_conv.size())
            if x.size(-1) == 1:
                x = self.relu(time_conv)
            else:
                # print("xx:", x.size(-1))
                x = self.relu(x[:, :, :, -time_conv.size(3):] + time_conv)  # B,F,N,T

            x = x + residual[:, :, :, -x.size(3):]
            x = x.data
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)
        # print("x:", x.shape)
        # print("skip:", self.skipE(x).shape)
        skip = self.skipE(x) + skip[:, :, :, -x.size(3):]
        # print("skip1:", skip.shape)
        # Output Module
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # print("x1:", x.shape)
        return x
