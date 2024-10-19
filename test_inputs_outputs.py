import torch
import os

from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_sequence
import utils

if __name__ == "__main__":
    patch_model_path = os.path.dirname(__file__)
    patch_model, agent_params, epoch = utils.load_pretrained_model(
        patch_model_path, "best")
    print("epoch ", epoch)

    model_params = {'min_level': 0}
    patch_size = agent_params['model_params']['img_size']  #
    stride_size = (68, 60, 48)
    invalid_margin = (7, 9, 8)
    # stride_size = (48, 48, 32)
    # invalid_margin = (14, 18, 16)
    device = "cuda:0"

    # [t,1,x,y,z]
    test_patch = torch.rand([5, 1, 96, 96, 80]).cuda()
    with torch.no_grad():
        packed_img_seq = pack_sequence([test_patch]).to(device)
        patch_pack_seq = PackedSequence(data=test_patch,
                                        batch_sizes=packed_img_seq.batch_sizes)
        _, _, patch_disp = patch_model(patch_pack_seq, **model_params)
        patch_disp = patch_disp.detach()
    print(patch_disp.shape)
