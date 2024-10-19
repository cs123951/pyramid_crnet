import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import filters
from layers import Conv, conv_disp, RNN_block


class RegistrationModel(nn.Module):

    def __init__(self,
                 device_ids,
                 input_device,
                 output_device,
                 img_size,
                 n_steps=3,
                 depthwise=False):
        super().__init__()
        self.device_ids = device_ids
        self.input_device = input_device
        self.output_device = output_device

        self.dim = len(img_size)
        self.img_size = img_size
        self.n_steps = n_steps
        self.depthwise = depthwise


class PyramidCommonModel(RegistrationModel):

    def __init__(self,
                 device_ids,
                 input_device,
                 output_device,
                 img_size,
                 enc_feats,
                 num_layers,
                 norm,
                 depthwise,
                 filter_name,
                 filter_params,
                 fp16=False,
                 use_rnn=True,
                 fuse_type='composite'):
        super().__init__(device_ids, input_device, output_device, img_size, 0,
                         depthwise)
        self.rnn_device = output_device
        self.disp_device = output_device
        self.norm = norm
        self.interp_mode = "trilinear" if self.dim == 3 else "bicubic"
        self.levels = len(enc_feats)
        self.num_layers = num_layers
        self.fp16 = fp16
        self.use_rnn = use_rnn
        self.fuse_type = fuse_type

        self.image_size_list = []
        self.encoder = []

        for idx in range(self.levels):
            rnn_size = self.get_rnn_size(idx)
            self.image_size_list.append(rnn_size)
            conv_blocks = self.get_enc_convs(idx, enc_feats)
            self.encoder.append(
                nn.Sequential(*conv_blocks).to(self.input_device))
        self.encoder = nn.ModuleList(self.encoder)

        if use_rnn:
            self.create_rnn_modules(enc_feats)
        else:
            self.create_cnn_modules(enc_feats)

        self.scale_factor = 2.0
        self.down_scale_factor = 1 / 2
        if depthwise:
            self.scale_factor = (2.0, 2.0, 1)
            self.down_scale_factor = (1 / 2, 1 / 2, 1)

        filter_params['device'] = output_device
        self.filter_name = filter_name
        self.filter = filters.get_filter_by_name(filter_name,
                                                 self.output_device,
                                                 filter_params)

    def calculate_enc_feats(self, packed_img_seq):
        batch_fix, batch_mov, disp_batch_sizes = utils.get_pair_from_packed_seq(
            packed_img_seq)

        batch_disp = None
        pyramid_fix_feat_list, pyramid_mov_feat_list = self.get_pyramid_feats(
            batch_fix, batch_mov)
        pyramid_warp_feat = pyramid_mov_feat_list[-1]
        return (batch_fix, batch_mov, pyramid_fix_feat_list,
                pyramid_mov_feat_list, pyramid_warp_feat, batch_disp,
                disp_batch_sizes)

    def calculate_composite_disp(self,
                                 batch_disp,
                                 disp_finer,
                                 cur_level,
                                 pyd_fix_feats,
                                 pyd_mov_feats,
                                 is_smooth=True):
        if batch_disp is None:
            batch_disp = disp_finer
        else:
            new_disp = disp_finer + utils.grid_sample_without_grid(
                batch_disp, disp_finer, has_normed=self.norm)
            batch_disp = self.fuse_disps(cur_level, batch_disp, new_disp,
                                         pyd_fix_feats[cur_level],
                                         pyd_mov_feats[cur_level])
        if is_smooth:
            if self.filter is not None:
                batch_disp = self.filter(batch_disp)

        return batch_disp

    def calculate_next_disp_feat(self, batch_disp, cur_level, pyd_mov_feats):
        if self.norm is not True:
            if self.depthwise:
                batch_disp[:, 0:2] = batch_disp[:, 0:2] * 2.0
            else:
                batch_disp = batch_disp * 2.0

        batch_disp = F.interpolate(batch_disp,
                                   size=self.image_size_list[cur_level - 1],
                                   mode='trilinear',
                                   align_corners=True)
        pyd_warp_feat = utils.grid_sample_without_grid(
            pyd_mov_feats[cur_level - 1], batch_disp, has_normed=self.norm)
        return batch_disp, pyd_warp_feat

    def create_rnn_modules(self, enc_feats):
        self.rnn_encoder = []
        self.forward_rnn = []
        self.backward_rnn = []
        self.rnn_decoder = []

        for idx in range(self.levels):
            rnn_conv = Conv(enc_feats[idx][-1] * 2,
                            enc_feats[idx][-1]).to(self.disp_device)
            self.rnn_encoder.append(
                nn.Sequential(rnn_conv).to(self.input_device))

            forward_rnn, backward_rnn = self.get_rnn_blocks(
                idx, enc_feats[idx])
            self.forward_rnn.append(forward_rnn)
            self.backward_rnn.append(backward_rnn)

            conv_blocks = [
                Conv(enc_feats[idx][-1] * 2,
                     enc_feats[idx][-1]).to(self.disp_device),
                conv_disp(enc_feats[idx][-1]).to(self.disp_device)
            ]
            self.rnn_decoder.append(
                nn.Sequential(*conv_blocks).to(self.input_device))

        self.rnn_encoder = nn.ModuleList(self.rnn_encoder)
        self.forward_rnn = nn.ModuleList(self.forward_rnn)
        self.backward_rnn = nn.ModuleList(self.backward_rnn)
        self.rnn_decoder = nn.ModuleList(self.rnn_decoder)

    def create_cnn_modules(self, enc_feats):
        self.disp_encoder = []
        for idx in range(self.levels):
            conv_blocks = [
                Conv(enc_feats[idx][-1] * 2,
                     enc_feats[idx][-1]).to(self.disp_device),
                Conv(enc_feats[idx][-1],
                     enc_feats[idx][-1]).to(self.disp_device),
                conv_disp(enc_feats[idx][-1]).to(self.disp_device)
            ]
            self.disp_encoder.append(
                nn.Sequential(*conv_blocks).to(self.input_device))
        self.disp_encoder = nn.ModuleList(self.disp_encoder)

    def get_rnn_size(self, idx):
        rnn_size = [self.img_size[jdx] // (2**idx) for jdx in range(self.dim)]
        if self.depthwise:
            rnn_size = [
                self.img_size[0] // (2**idx),
                self.img_size[1] // (2**idx),
                self.img_size[2],
            ]
        return rnn_size

    def get_enc_convs(self, idx, feats):
        conv_blocks = []
        if idx > 0:
            if self.depthwise:
                conv_blocks = [nn.MaxPool3d(kernel_size=(2, 2, 1))]
            else:
                conv_blocks = [nn.MaxPool3d(kernel_size=2)]
            conv_blocks.append(Conv(feats[idx - 1][-1], feats[idx][0]))
        else:
            conv_blocks.append(Conv(1, feats[idx][0]))
        for jdx in range(len(feats[idx]) - 1):
            conv_blocks.append(Conv(feats[idx][jdx], feats[idx][jdx + 1]))
        return conv_blocks

    def fuse_disps(self, idx, disp1, disp_composite, fix, mov):
        if self.fuse_type == 'composite':
            return disp_composite
        elif self.fuse_type == 'weighted':
            return self.fuse_weighted_disps(idx, disp1, disp_composite, fix,
                                            mov)
        elif self.fuse_type == 'cnn':
            return self.fuse_cnn_disps(idx, disp1, disp_composite, fix, mov)
        elif self.fuse_type == 'ncc':
            return self.fuse_ncc_disps(disp1, disp_composite, fix, mov)
        elif self.fuse_type == 'dotproduct':
            return self.fuse_dotproduct_disps(idx, disp1, disp_composite, fix,
                                              mov)
        elif self.fuse_type == 'refine':
            # return self.refine_disp(idx, disp_composite, fix, mov)
            return self.refinedisp_module[idx](self.fuse_encoder[idx],
                                               disp_composite, fix, mov)
        elif self.fuse_type == 'correlation':
            return self.fuse_disp_by_correlation(disp_composite, fix, mov)

    def get_rnn_blocks(self, idx, enc_idx):
        hidden_dim_list = [enc_idx[-1]] * self.num_layers
        kernel_size_list = [3] * self.num_layers
        forward_rnn = RNN_block(img_size=self.image_size_list[idx],
                                hidden_dim_list=hidden_dim_list,
                                kernel_size_list=kernel_size_list,
                                num_layers=self.num_layers,
                                input_dim=enc_idx[-1],
                                fp16=self.fp16).to(self.rnn_device)
        backward_rnn = RNN_block(img_size=self.image_size_list[idx],
                                 hidden_dim_list=hidden_dim_list,
                                 kernel_size_list=kernel_size_list,
                                 num_layers=self.num_layers,
                                 input_dim=enc_idx[-1],
                                 fp16=self.fp16).to(self.rnn_device)
        return forward_rnn, backward_rnn

    def get_pyramid_feats(self, batch_fix, batch_mov):
        pyramid_fix = batch_fix
        pyramid_mov = batch_mov
        pyramid_fix_feat_list = []
        pyramid_mov_feat_list = []
        for idx in range(self.levels):
            pyramid_fix = self.encoder[idx](pyramid_fix)
            pyramid_mov = self.encoder[idx](pyramid_mov)
            pyramid_fix_feat_list.append(pyramid_fix)
            pyramid_mov_feat_list.append(pyramid_mov)
            # if self.train_sep:
            #     # pyramid_fix = batch_fix = self.gaussian_filter(batch_fix)[:, :, ::2, ::2, ::2]
            #     # pyramid_mov = batch_mov = self.gaussian_filter(batch_mov)[:, :, ::2, ::2, ::2]
            #     pyramid_fix = batch_fix = batch_fix[:, :, ::2, ::2, ::2]
            #     pyramid_mov = batch_mov = batch_mov[:, :, ::2, ::2, ::2]
        return pyramid_fix_feat_list, pyramid_mov_feat_list

    def bidirectional_rnn(self, fix_feature, warp_feature, cur_level,
                          disp_batch_sizes):
        x = torch.cat([fix_feature, warp_feature], dim=1)
        x = self.rnn_encoder[cur_level](x)
        rev_enc_feat = utils.reverse_packed_sequence(x, disp_batch_sizes)
        for_feat = self.forward_rnn[cur_level]([x, disp_batch_sizes])
        back_feat = self.backward_rnn[cur_level](
            [rev_enc_feat, disp_batch_sizes])
        rev_back_feat = utils.reverse_packed_sequence(back_feat,
                                                      disp_batch_sizes)

        rnn_feat = torch.cat([for_feat, rev_back_feat],
                             dim=1)  # channel is down_ch_list[1]*2
        rnn_disp = self.rnn_decoder[cur_level](rnn_feat)
        return rnn_disp

    def filter_disp(self, batch_mov, batch_disp):
        if self.filter is not None:
            if self.filter_name.startswith('guided'):
                batch_disp = self.filter(batch_mov, batch_disp)
            else:
                batch_disp = self.filter(batch_disp)
        return batch_disp


class PyramidCRNetv1(PyramidCommonModel):

    def __init__(self,
                 device_ids,
                 input_device,
                 output_device,
                 img_size,
                 enc_feats,
                 num_layers,
                 norm=False,
                 depthwise=False,
                 fp16=False,
                 filter_name="gaussian",
                 filter_params={"kernel_size": [5, 5, 3]},
                 fuse_type='composite'):
        super().__init__(device_ids,
                         input_device,
                         output_device,
                         img_size,
                         enc_feats,
                         num_layers,
                         norm,
                         depthwise,
                         filter_name,
                         filter_params,
                         fp16,
                         use_rnn=True,
                         fuse_type=fuse_type)

    def forward(self, packed_img_seq, min_level=0):
        # 1. extract features
        batch_fix, batch_mov, pyd_fix_feats, pyd_mov_feats, pyd_warp_feat, batch_disp, \
            disp_batch_sizes = self.calculate_enc_feats(packed_img_seq)

        for cur_level in range(self.levels - 1, -1, -1):
            disp_finer = self.bidirectional_rnn(pyd_fix_feats[cur_level],
                                                pyd_warp_feat, cur_level,
                                                disp_batch_sizes)
            batch_disp = self.calculate_composite_disp(batch_disp,
                                                       disp_finer,
                                                       cur_level,
                                                       pyd_fix_feats,
                                                       pyd_mov_feats,
                                                       is_smooth=False)

            if min_level == cur_level:
                if self.norm is not True:
                    batch_disp = batch_disp * (2.0**min_level)
                if min_level > 0:
                    batch_disp = F.interpolate(batch_disp,
                                               size=self.image_size_list[0],
                                               mode='trilinear',
                                               align_corners=True)
                if min_level == 0:
                    batch_disp = self.filter_disp(batch_mov, batch_disp)
                return batch_fix, batch_mov, batch_disp

            if cur_level > 0:
                batch_disp, pyd_warp_feat = self.calculate_next_disp_feat(
                    batch_disp, cur_level, pyd_mov_feats)

        return None
