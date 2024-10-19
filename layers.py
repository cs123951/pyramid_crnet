import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import Variable

import utils


class Add(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, xs, weights):
        outputs = 0
        for idx in range(len(xs)):
            outputs += xs[idx] * weights[idx]
        return outputs


class Conv(nn.Module):
    """
    it includes: convolution, batchnorm, relu, and pool(if is downsampled)
    The kernel size of convolution layer is constantly 3.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding_size=1,
                 padding="zeros",
                 stride_size=1,
                 dilation=1,
                 down=False,
                 up=False,
                 dim=3,
                 depthwise=False,
                 pointwise=False,
                 bn=True,
                 is_conv=True,
                 bias=True,
                 activation=nn.LeakyReLU(0.2, inplace=True),
                 dropout=-1):
        super(Conv, self).__init__()
        self.down = down
        self.up = up
        assert (self.down and self.up
                ) is False, "can not conduct up and down at the same time"

        self.dim = dim
        self.kernel_size = kernel_size
        self.pool_size = 2
        self.padding_size = padding_size
        self.stride_size = stride_size
        self.dilation = dilation
        self.bias = bias
        self.dropout = dropout
        self.is_conv = is_conv

        # only 3D image has depthwise
        if depthwise:
            self.kernel_size = (3, 3, 1)
            self.padding_size = (1, 1, 0)
            if self.down or self.up:
                self.pool_size = (2, 2, 1)
        if pointwise:
            self.point_kernel_size = (1, 1, 3)
            self.point_padding_size = (0, 0, 1)
            self.pool_size = 2
            self.down = True

        module_list = []
        if is_conv:
            conv_layer = getattr(nn,
                                 'Conv%dd' % dim)(in_channels,
                                                  out_channels,
                                                  kernel_size=self.kernel_size,
                                                  padding=self.padding_size,
                                                  stride=self.stride_size,
                                                  bias=self.bias,
                                                  dilation=self.dilation)
            module_list.append(conv_layer)

        if pointwise:
            point_conv_layer = getattr(nn, 'Conv%dd' % dim)(
                out_channels,
                out_channels,
                kernel_size=self.point_kernel_size,
                padding=self.point_padding_size,
                stride=self.stride_size,
                bias=self.bias,
                dilation=self.dilation)
            module_list.append(point_conv_layer)

        if bn:
            bn_layer = getattr(nn, 'BatchNorm%dd' % dim)(out_channels)
            module_list.append(bn_layer)

        if activation is not None:
            module_list.append(activation)

        if down:
            self.pool = getattr(nn, 'AvgPool%dd' % dim)(self.pool_size)

        if up:
            self.pool = nn.Upsample(scale_factor=self.pool_size,
                                    align_corners=True)

        if is_conv:
            self.conv = nn.Sequential(*module_list)
            self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, inp_tensor):
        if self.is_conv:
            inp_tensor = self.conv(inp_tensor)
        if self.dropout > 0:
            inp_tensor = getattr(nnf, 'dropout%dd' % self.dim)(inp_tensor,
                                                               p=self.dropout,
                                                               training=True)
        if self.down or self.up:
            inp_tensor = self.pool(inp_tensor)
        return inp_tensor


class RefineDisp(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, fuse_encoder_idx, disp, fix, mov):
        warp = utils.grid_sample_without_grid(mov, disp)  # [b, c, x, y, z]
        alpha = fuse_encoder_idx(fix - warp)
        return alpha * disp + disp


class conv_disp(nn.Module):

    def __init__(self, inChan, outChan=3, kernel_size=3, dim=3):
        super(conv_disp, self).__init__()
        self.dim = dim

        if self.dim == 3:
            self.conv = nn.Conv3d(inChan,
                                  outChan,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  padding=int(kernel_size // 2),
                                  bias=True)
        elif self.dim == 2:
            self.conv = nn.Conv2d(inChan,
                                  outChan,
                                  kernel_size=kernel_size,
                                  stride=1,
                                  padding=int(kernel_size // 2),
                                  bias=True)

        self.conv.weight.data.normal_(mean=0, std=1e-5)
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()

    def forward(self, inp_tensor):
        # print("x.shape: ", x.shape, type(x), type(self.conv.weight))
        inp_tensor = self.conv(inp_tensor)
        return inp_tensor


class RNN_block(nn.Module):

    def __init__(self,
                 img_size,
                 input_dim,
                 hidden_dim_list,
                 kernel_size_list,
                 num_layers,
                 batch_first=True,
                 bias=True,
                 rnn_cell='ConvGRUCell',
                 depthwise=False,
                 dropout=-1,
                 conv_type='conv',
                 cell_params=dict(),
                 fp16=False):
        super(RNN_block, self).__init__()

        self.img_size = img_size
        self.dim = len(img_size)
        self.input_dim = input_dim
        self.hidden_dim_list = hidden_dim_list
        self.kernel_size_list = kernel_size_list
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.dropput = dropout
        self.rnn_cell = rnn_cell
        self.cell_params = cell_params
        Cell = ConvGRUCell

        cell_list = []
        for i in range(0, self.num_layers):
            # for [1,] layer, the input is the last hidden output
            cur_input_dim = input_dim if i == 0 else hidden_dim_list[i - 1]
            cell_list.append(
                Cell(input_size=img_size,
                     input_dim=cur_input_dim,
                     hidden_dim=self.hidden_dim_list[i],
                     kernel_size=self.kernel_size_list[i],
                     bias=self.bias,
                     depthwise=depthwise,
                     conv_type=conv_type,
                     fp16=fp16,
                     **self.cell_params))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, inputs):
        """"
        batch_input: (b*t, c, x, y, z), a list of length b

        """
        batch_input, batch_sizes = inputs
        seq_len = len(batch_sizes)
        num_cases = batch_sizes[0]
        device = batch_input.device
        init_hidden_size = self.hidden_dim_list[0]

        if self.rnn_cell == 'ConvLSTMCell':
            cur_state = []  # num_layers, 2, batch, h, x, y, z
            for _ in range(self.num_layers):
                h_unit = Variable(
                    torch.zeros(num_cases, init_hidden_size,
                                *self.img_size)).to(device)
                c_unit = Variable(
                    torch.zeros(num_cases, init_hidden_size,
                                *self.img_size)).to(device)
                cur_state.append([h_unit, c_unit])
        elif self.rnn_cell in ['ConvGRUCell', 'ProposedCell']:
            # num_layers, [batch_t, h, x, y, z]
            cur_state = []
            for _ in range(self.num_layers):
                cur_state.append(
                    Variable(
                        torch.zeros(num_cases,
                                    init_hidden_size,
                                    *self.img_size,
                                    dtype=torch.float16)).to(device))

        output_list = []
        begin_idx = 0
        for t in range(seq_len):
            num_batch = batch_sizes[t].item()
            # batch_input_t is [batch_size_t,c,x,y,z] #  -> [bs, h_dim, *conv_size]
            batch_input_t = batch_input[begin_idx:begin_idx + num_batch]

            if self.rnn_cell == 'ConvLSTMCell':
                cur_state_t = []  # num_layers, 2, batch, h, x, y, z
                for ly in range(self.num_layers):
                    cur_state_t.append([
                        cur_state[ly][0][:num_batch],
                        cur_state[ly][1][:num_batch]
                    ])
            elif self.rnn_cell in ['ConvGRUCell', 'ProposedCell']:
                # cur_state is {num_layers, [batch, h, x, y, z]}
                # cur_state_t is { num_layers, batch_t, h, x, y, z}
                cur_state_t = [
                    cur_state[ly][:num_batch] for ly in range(self.num_layers)
                ]

            cur_t_layer_i = batch_input_t  # initialize current input, [batch, c, x, y, z]
            for layer_idx in range(self.num_layers):
                # print("t, ", t, " layer_idx ", layer_idx, batch_input_t.shape, cur_state_t[0].shape)
                # input current hidden and cell state, then compute the next hidden and cell state.
                cell = self.cell_list[layer_idx]

                cell_output = cell(input_tensor=cur_t_layer_i,
                                   cur_state=cur_state_t[layer_idx])
                if self.rnn_cell == 'ConvLSTMCell':
                    cur_t_layer_i = cell_output[0]
                    cur_state[layer_idx][0] = cell_output[0]
                    cur_state[layer_idx][1] = cell_output[1]
                elif self.rnn_cell in ['ConvGRUCell', 'ProposedCell']:
                    cur_t_layer_i = cell_output
                    cur_state[layer_idx] = cell_output
                else:
                    print("error")
                    return None
                if self.dropput > 0:
                    cur_t_layer_i = getattr(nnf, "dropout%dd" % self.dim)(
                        cur_t_layer_i, p=self.dropput, training=True)
            begin_idx += num_batch

            if self.rnn_cell == 'ConvLSTMCell':
                output_list.append(cell_output[0])  # output h_state
            elif self.rnn_cell in ['ConvGRUCell', 'ProposedCell']:
                output_list.append(cell_output)

        return torch.cat(output_list, dim=0)


class ConvGRUCell(nn.Module):

    def __init__(self,
                 input_size,
                 input_dim=2,
                 hidden_dim=16,
                 kernel_size=3,
                 bias=True,
                 depthwise=False,
                 conv_type='conv',
                 fp16=False):
        super(ConvGRUCell, self).__init__()

        self.padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.dim = len(input_size)

        kernel_size_list = tuple([kernel_size] * self.dim)
        if depthwise:
            kernel_size_list = tuple([kernel_size, kernel_size,
                                      1])  # only in 3d case
            self.padding = tuple([kernel_size // 2, kernel_size // 2, 0])

        if conv_type == 'conv':
            conv_block = getattr(nn, 'Conv%dd' % self.dim)

        if self.dim == 3:
            self.x_size, self.y_size, self.z_size = input_size
        elif self.dim == 2:
            self.x_size, self.y_size = input_size

        if fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.conv_gates = conv_block(
            in_channels=input_dim + hidden_dim,
            out_channels=2 *
            self.hidden_dim,  # for update_gate,reset_gate respectively
            kernel_size=kernel_size_list,
            padding=self.padding,
            bias=self.bias)

        self.conv_can = conv_block(
            in_channels=input_dim + hidden_dim,
            out_channels=self.hidden_dim,  # for candidate neural memory
            kernel_size=kernel_size_list,
            padding=self.padding,
            bias=self.bias)

    def forward(self, input_tensor, cur_state=None):
        # input_tensor: [batch, channel, x,y,z]
        # h_cur: [batch, channel, x, y, z]
        h_cur = cur_state
        if h_cur is None:
            if self.dim == 3:
                h_cur = Variable(torch.zeros(input_tensor.shape[0],
                                             self.hidden_dim, self.x_size,
                                             self.y_size, self.z_size),
                                 dtype=self.dtype).to(input_tensor.device)
            elif self.dim == 2:
                h_cur = Variable(
                    torch.zeros(input_tensor.shape[0],
                                self.hidden_dim,
                                self.x_size,
                                self.y_size,
                                dtype=self.dtype)).to(input_tensor.device)
        combined = torch.cat([input_tensor, h_cur],
                             dim=1)  # [batch, hidden_dim+input_dim, x, y, z]
        combined_conv = self.conv_gates(
            combined)  # [batch, 2*hidden_dim, x, y, z]
        gamma, beta = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )  # [batch, hidden_dim, x, y, z], [batch, hidden_dim, x, y, z]
        reset_gate = torch.sigmoid(gamma)  # R_t  [batch, hidden_dim, x, y, z]
        update_gate = torch.sigmoid(beta)  # Z_t  [batch, hidden_dim, x, y, z]

        combined = torch.cat(
            [input_tensor, reset_gate * h_cur], dim=1
        )  # X_t, R_t \circ H_(t-1)   [batch, input_dim+hidden_dim, x, y, z]
        cc_cnm = self.conv_can(combined)  # [batch, hidden_dim, x, y, z]
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate
                  ) * h_cur + update_gate * cnm  # [batch, hidden_dim, x, y, z]

        return h_next
