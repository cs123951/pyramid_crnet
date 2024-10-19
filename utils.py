import torch
from torch.nn.utils.rnn import pack_sequence, PackedSequence
import torch.nn.functional as F
import importlib
import os
import json
from einops import rearrange


def compute_neighbor_grid(max_disp):
    xx, yy, zz = torch.meshgrid([
        torch.linspace(-max_disp, max_disp, steps=1 + 2 * max_disp),
        torch.linspace(-max_disp, max_disp, steps=1 + 2 * max_disp),
        torch.linspace(-max_disp, max_disp, steps=1 + 2 * max_disp)
    ],
                                indexing='ij')
    grid = torch.stack((xx, yy, zz), 0).unsqueeze(0)  # [b=1, 3, x,y,z]
    return grid


def reverse_packed_sequence(packed_seq_tensor, batch_sizes):
    """_summary_

    Args:
        packed_seq_tensor (torch.tensor): The first dim includes [].
        batch_sizes (_type_): _description_

    Returns:
        totch.tensor: _description_
    """
    list_tensor = unpack_sequence([packed_seq_tensor, batch_sizes])
    rev_list_tensor = [torch.flip(tensor_tx, [0]) for tensor_tx in list_tensor]
    rev_packed_seq = pack_sequence(rev_list_tensor)
    return rev_packed_seq.data


def unpack_sequence(packed_seq):
    """Unpacks PackedSequence into a list of variable length Tensors

    Args:
        packed_sequence (PackedSequence or list):
        (1) PackedSequence
                PackedSequence has four variables:
                    data (Tensor)
                    batch_sizes (Tensor)
        (2) list is [tensor, batch_sizes]
        batch_sizes (torch.tensor): Tensor of integers holding information about the batch size at each sequence step

    Returns:
        ori_seq (list): in the form of [[t1,*],[t2,*],...]
    """
    # maker sure that packed_sequence is sorted [b*t,c,x,y,z]
    if isinstance(packed_seq, PackedSequence):
        batch_sizes = packed_seq.batch_sizes
        data = packed_seq.data
    elif isinstance(packed_seq, list):
        data, batch_sizes = packed_seq

    device = data.device
    indices = {}
    for idx in range(batch_sizes[0]):
        indices[idx] = [idx]
    count = batch_sizes[0].item()

    for bs in batch_sizes[1:]:
        for idx in range(bs):
            indices[idx].append(count)
            count += 1
    ori_seq = []  # b*[t,*]

    for kdx in indices.keys():
        ori_seq.append(
            torch.index_select(data, 0,
                               torch.tensor(indices[kdx], device=device)))

    # b*[t,*]
    return ori_seq


def get_pair_from_packed_seq(packed_seq, frame='lag'):
    """_summary_

    Args:
        packed_seq (torch.nn.utils.rnn.PackedSequence): _description_

    Returns:
        _type_: _description_
    """
    fix_indices = []
    mov_indices = []

    batch_sizes = packed_seq.batch_sizes
    indices = list(range(batch_sizes[0]))
    packed_seq = packed_seq.data
    device = packed_seq.device

    mov_start = batch_sizes[0].item()
    fix_start = 0
    for bs in batch_sizes[1:]:
        bs = bs.item()
        if frame == 'lag':
            fix_indices.extend(indices[fix_start:fix_start + bs])
        elif frame == 'euler':
            fix_indices.extend(list(range(fix_start, fix_start + bs)))
            fix_start += bs
        mov_indices.extend(list(range(mov_start, mov_start + bs)))
        mov_start += bs
    batch_fix = torch.index_select(packed_seq, 0,
                                   torch.tensor(fix_indices, device=device))
    batch_mov = torch.index_select(packed_seq, 0,
                                   torch.tensor(mov_indices, device=device))
    return batch_fix, batch_mov, batch_sizes[1:]  # [b, 2, x, y, z]


def get_class_by_name(mc_name, anchor_name):
    if anchor_name == '__main__' or not mc_name.startswith('.'):
        package = None
    else:
        package = anchor_name.rsplit('.', 1)[0]

    module_name, class_name = mc_name.rsplit('.', 1)
    # import sys
    # print(module_name)
    # print(package)
    # print(sys.path)

    file_module = importlib.import_module(module_name, package)
    class_module = getattr(file_module, class_name)
    return class_module


def parse_json_file(json_name="config.json", configs_root=""):
    json_file = os.path.join(configs_root, json_name)
    with open(json_file, 'r') as f:
        args = json.load(f)
    return args


def load_model(model, weight_file_path, ddp=True):
    weights = torch.load(weight_file_path)
    if ddp:
        model.module.load_state_dict(weights['state_dict'])
    else:
        model.load_state_dict(weights['state_dict'])

    start_epoch = weights['epoch']
    if 'valid_loss' in weights:
        best_loss = weights['valid_loss']
    else:
        best_loss = weights['val_loss']
    return model, start_epoch, best_loss


def load_pretrained_model(model_path, pth='best'):
    agent_params = parse_json_file(json_name="config_demo.json",
                                   configs_root=model_path)
    model_class = get_class_by_name('models.' + agent_params["model_name"],
                                    __name__)(**agent_params['model_params'])
    # weight_file_path = os.path.join(model_path, pth + ".pth")
    random_init_model = model_class

    # pretrained_model, epoch, _ = load_model(model_class,
    #                                         weight_file_path,
    #                                         ddp=False)
    # print("load epoch ", epoch)
    # return pretrained_model, agent_params, epoch
    return random_init_model, agent_params, 0


def grid_sample_without_grid(
    inputs,
    displacement,
    regular_grid=None,
    padding_mode="border",
    interp_mode="bilinear",
    align_corners=True,
    has_normed=True,
):
    """
    no grid but flow
    :param inputs: [batch, n, x, y, z]
    :param displacement: [batch, 3, x, y, z]
    :param regular_grid: [batch, 3, x, y, z]
    :param padding_mode:
    :param interp_mode:
    :return: [N,C,x,y,z]
    """
    dim = len(inputs.shape) - 2
    if regular_grid is None:
        grid_size = displacement.shape[2:]
        batch_size = len(inputs)
        device = displacement.device
        regular_grid = create_batch_regular_grid(img_size=grid_size,
                                                 batch_size=batch_size,
                                                 device=device,
                                                 to_norm=has_normed)

    deformation_field = regular_grid + displacement
    if not has_normed:
        deformation_field = norm_deformation(deformation_field)

    if dim == 3:
        deformation_field_ch_last = rearrange(deformation_field,
                                              "b c x y z -> b x y z c")
    elif dim == 2:
        deformation_field_ch_last = rearrange(deformation_field,
                                              "b c x y -> b x y c")

    output = grid_sample_with_grid(
        inputs,
        deformation_field_ch_last,
        padding_mode=padding_mode,
        interp_mode=interp_mode,
        align_corners=align_corners,
    )
    return output


def grid_sample_with_grid(
    inputs,
    deformation_field,
    padding_mode="border",
    interp_mode="bilinear",
    align_corners=True,
):
    """
    :param inputs: [batch, 1, x,y,z]
    :param deformation_field:  [batch, x,y,z, 3]
    :param padding_mode:
    :param interp_mode:
    :return:
    """
    grid_rev = torch.flip(deformation_field, [-1])  # flip the dim
    output_tensor = F.grid_sample(
        inputs,
        grid_rev,
        padding_mode=padding_mode,
        mode=interp_mode,
        align_corners=align_corners,
    )
    # output_tensor is [N,C,x,y,z]
    return output_tensor


def create_batch_regular_grid(img_size,
                              batch_size=1,
                              device="cpu",
                              to_norm=True):
    """

    :param img_size:
    :param batch_size:
    :param device:
    :return: regular grid [batch, 3, x,y,z]
    """
    dim = len(img_size)
    if dim == 3:
        D, H, W = img_size

        x_range = torch.arange(0, D, device=device)
        y_range = torch.arange(0, H, device=device)
        z_range = torch.arange(0, W, device=device)

        if to_norm:
            x_range = x_range * 2.0 / (D - 1.0) - 1.0
            y_range = y_range * 2.0 / (H - 1.0) - 1.0
            z_range = z_range * 2.0 / (W - 1.0) - 1.0

        regular_grid_list = torch.meshgrid(x_range,
                                           y_range,
                                           z_range,
                                           indexing="ij")  # 3*[x,y,z]
        regular_grid = torch.stack(regular_grid_list, dim=0).unsqueeze(
            0)  # [3,x,y,z] -> [1,3,x,y,z]
        batch_regular_grid = regular_grid.repeat(batch_size, 1, 1, 1, 1)
    elif dim == 2:
        H, W = img_size

        y_range = torch.arange(0, H, device=device)
        z_range = torch.arange(0, W, device=device)

        if to_norm:
            y_range = y_range * 2.0 / (H - 1.0) - 1.0
            z_range = z_range * 2.0 / (W - 1.0) - 1.0

        regular_grid_list = torch.meshgrid(y_range, z_range,
                                           indexing="ij")  # 3*[x,y,z]
        regular_grid = torch.stack(regular_grid_list, dim=0).unsqueeze(0)
        batch_regular_grid = regular_grid.repeat(batch_size, 1, 1, 1)

    return batch_regular_grid
    # return batch_regular_grid.half()


def norm_deformation(deformation, img_size=None):
    if img_size is None:
        img_size = deformation.shape[2:]

    dim = len(img_size)
    new_deformation = deformation.clone()

    for idx in range(dim):
        new_deformation[:, idx] = (new_deformation[:, idx] * 2.0 /
                                   (img_size[idx] - 1.0) - 1.0)

    return new_deformation
