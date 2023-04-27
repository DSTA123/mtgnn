from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
import math


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        # print("cout:",cout)
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))
            # print("kern:",kern)
            # print("self:",self)

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
            # print("input:",input.shape)
            # print("xx:",x[i].shape)
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
            # print("xi:", x[i].shape)
        x = torch.cat(x,dim=1)
        # print("x:", x.shape)
        return x


class GateDilatedConv(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size):
        super(GateDilatedConv, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_strides))

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(x_p, self.sigmoid(x_q))
        return x_gtu


class GraphLearn(torch.nn.Module):
    """
    Graph Learning Modoel for AdapGL.

    Args:
        num_nodes: The number of nodes.
        init_feature_num: The initial feature number (< num_nodes).
    """
    def __init__(self, num_nodes, init_feature_num):
        super(GraphLearn, self).__init__()
        self.epsilon = 1 / num_nodes * 0.5
        self.beta = torch.nn.Parameter(
            torch.rand(num_nodes, dtype=torch.float32),
            requires_grad=True
        )

        self.w1 = torch.nn.Parameter(
            torch.zeros((num_nodes, init_feature_num), dtype=torch.float32),
            requires_grad=True
        )
        self.w2 = torch.nn.Parameter(
            torch.zeros((num_nodes, init_feature_num), dtype=torch.float32),
            requires_grad=True
        )

        self.attn = torch.nn.Conv2d(2, 1, kernel_size=1)

        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, adj_mx):
        new_adj_mx = torch.mm(self.w1, self.w2.T) - torch.mm(self.w2, self.w1.T)
        new_adj_mx = torch.relu(new_adj_mx + torch.diag(self.beta))
        attn = torch.sigmoid(self.attn(torch.stack((new_adj_mx, adj_mx), dim=0).unsqueeze(dim=0)).squeeze())
        new_adj_mx = attn * new_adj_mx + (1. - attn) * adj_mx
        d = new_adj_mx.sum(dim=1) ** (-0.5)
        new_adj_mx = torch.relu(d.view(-1, 1) * new_adj_mx * d - self.epsilon)
        d = new_adj_mx.sum(dim=1) ** (-0.5)
        new_adj_mx = d.view(-1, 1) * new_adj_mx * d
        return new_adj_mx

class GraphConv(torch.nn.Module):
    r"""
    Graph Convolution with self feature modeling.

    Args:
        f_in: input size.
        num_cheb_filter: output size.
        conv_type:
            gcn: :math:`AHW`,
            cheb: :math:``T_k(A)HW`.
        activation: default relu.
    """
    def __init__(self, f_in, num_cheb_filter, conv_type=None, **kwargs):
        super(GraphConv, self).__init__()
        self.K = kwargs.get('K', 3) if conv_type == 'cheb' else 1
        self.with_self = kwargs.get('with_self', True)
        self.w_conv = torch.nn.Linear(f_in * self.K, num_cheb_filter, bias=False)
        if self.with_self:
            self.w_self = torch.nn.Linear(f_in, num_cheb_filter)
        self.conv_type = conv_type
        self.activation = kwargs.get('activation', torch.relu)

    def cheb_conv(self, x, adj_mx):
        bs, num_nodes, _ = x.size()

        if adj_mx.dim() == 3:
            h = x.unsqueeze(dim=1)
            h = torch.matmul(adj_mx, h).transpose(1, 2).reshape(bs, num_nodes, -1)
        else:
            h_list = [x, torch.matmul(adj_mx, x)]
            for _ in range(2, self.K):
                h_list.append(2 * torch.matmul(adj_mx, h_list[-1]) - h_list[-2])
            h = torch.cat(h_list, dim=-1)

        h = self.w_conv(h)
        if self.with_self:
            h += self.w_self(x)
        if self.activation is not None:
            h = self.activation(h)
        return h

    def gcn_conv(self, x, adj_mx):
        h = torch.matmul(adj_mx, x)
        h = self.w_conv(h)
        if self.with_self:
            h += self.w_self(x)
        if self.activation is not None:
            h = self.activation(h)
        return h

    def forward(self, x, adj_mx):
        self.conv_func = self.cheb_conv if self.conv_type == 'cheb' else self.gcn_conv
        return self.conv_func(x, adj_mx)

class TemporalAttention(torch.nn.Module):
    """ Compute Temporal attention scores.
0
    Args:
        num_nodes: Number of vertices.
        f_in: Number of features.
        c_in: Number of time steps.

    Shape:
        - Input: :math:`(batch\_size, c_{in}, num\_nodes, f_{in})`
        - Output: :math:`(batch\_size, c_in, c_in)`.
    """
    def __init__(self, num_nodes, f_in, c_in):
        super(TemporalAttention, self).__init__()

        self.w1 = torch.nn.Parameter(torch.randn(num_nodes, dtype=torch.float32), requires_grad=True)
        self.w2 = torch.nn.Linear(f_in, num_nodes, bias=False)
        self.w3 = torch.nn.Parameter(torch.randn(f_in, dtype=torch.float32), requires_grad=True)
        self.be = torch.nn.Parameter(torch.randn(1, c_in, c_in, dtype=torch.float32), requires_grad=True)
        self.ve = torch.nn.Parameter(
            torch.zeros(c_in, c_in, dtype=torch.float32),
            requires_grad=True
        )

        torch.nn.init.kaiming_uniform_(self.ve, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.be, a=math.sqrt(5))
        torch.nn.init.uniform_(self.w1)
        torch.nn.init.uniform_(self.w3)

    def forward(self, x):
        # print("x:", x.size())
        y1 = torch.matmul(x.transpose(2, 3), self.w1)
        # print("y1:",y1.size())
        y1 = self.w2(y1)

        y2 = torch.matmul(x, self.w3).transpose(1, 2)

        product = torch.matmul(y1, y2)
        E = torch.matmul(self.ve, torch.sigmoid(product + self.be))
        E = F.softmax(E, dim=-1)
        # print("E:", E.size())
        return E

class SpatialAttention(torch.nn.Module):
    """ Compute Spatial attention scores.

    Args:
        num_nodes: Number of nodes.
        f_in: Number of features.
        c_in: Number of time steps.

    Shape:
        - Input: :math:`(batch\_size, c_{in}, num\_nodes, f_{in})`
        - Output: :math:`(batch\_size, num\_nodes, num\_nodes)`.
    """
    def __init__(self, num_nodes, f_in, c_in):
        super(SpatialAttention, self).__init__()

        self.w1 = torch.nn.Conv2d(c_in, 1, 1, bias=False)
        self.w2 = torch.nn.Linear(f_in, c_in, bias=False)
        self.w3 = torch.nn.Parameter(
            torch.randn(f_in, dtype=torch.float32),
            requires_grad=True
        )
        self.vs = torch.nn.Parameter(
            torch.randn(num_nodes, num_nodes, dtype=torch.float32),
            requires_grad=True
        )

        self.bs = torch.nn.Parameter(
            torch.randn(num_nodes, num_nodes, dtype=torch.float32),
            requires_grad=True
        )

        torch.nn.init.kaiming_uniform_(self.vs, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.bs, a =math.sqrt(5))
        torch.nn.init.uniform_(self.w3)

    def forward(self, x):
        # print("x:", x.size())
        y1 = self.w1(x).squeeze(dim=1)
        y1 = self.w2(y1)
        y2 = torch.matmul(x, self.w3)

        product = torch.matmul(y1, y2)
        y = torch.matmul(self.vs, torch.sigmoid(product + self.bs))
        y = F.softmax(y, dim=-1)
        return y


class ChebConv(torch.nn.Module):
    """
    Graph Convolution with Chebyshev polynominals.

    Args:
        - input_feature: Dimension of input features.
        - out_feature: Dimension of output features.
        - adj_mx: Adjacent matrix with shape :math:`(K, num\_nodes, num\_nodes)` followed by
          Kth Chebyshev polynominals, where :math:`T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x-2)` with
          :math:`T_0(x)=1, T_1(x) = x`.

    Shape:
        - Input:
            x: :math:`(batch\_size, c_in, num\_nodes, f_in)`.
            spatial_att: :math:`(batch\_size, num\_nodes, num\_nodes)`
        - Output:
            :math:`(batch_size, c_in, num_\nodes, f_out).
    """
    def __init__(self, input_feature, out_feature, adj_mx):
        super(ChebConv, self).__init__()

        self.adj_mx = adj_mx
        self.w = torch.nn.Linear(input_feature, out_feature, bias=False)

    def forward(self, x, spatial_att):
        b, c_in, num_nodes, _ = x.size()

        outputs = []
        adj = spatial_att.unsqueeze(dim=1) * self.adj_mx
        for i in range(c_in):
            x1 = x[:, i].unsqueeze(dim=1)
            y = torch.matmul(adj, x1).transpose(1, 2).reshape(b, num_nodes, -1)
            y = torch.relu(self.w(y))
            outputs.append(y)
        return torch.stack(outputs, dim=1)

class ChannelAttention(torch.nn.Module):
    def __init__(self, c_in):
        super(ChannelAttention, self).__init__()
        self.r = 0.5
        hidden_size = int(c_in / self.r)
        self.w1 = torch.nn.Linear(c_in, hidden_size, bias=False)
        self.w2 = torch.nn.Linear(hidden_size, c_in, bias=False)

    def forward(self, x):
        y = x.mean(dim=(-1, -2))
        y = torch.sigmoid(self.w2(torch.relu(self.w1(y))))
        return y.unsqueeze(dim=-1)

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho

class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)


    def forward(self,x):
        #adj = adj + torch.eye(adj.size(0)).to(x.device)
        #d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)


        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2



class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        # print("cout:",cout)
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))
            # print("kern:",kern)
            # print("self:",self)

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
            # print("input:",input.shape)
            # print("xx:",x[i].shape)
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
            # print("xi:", x[i].shape)
        x = torch.cat(x,dim=1)
        # print("x:", x.shape)
        return x


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj


class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
