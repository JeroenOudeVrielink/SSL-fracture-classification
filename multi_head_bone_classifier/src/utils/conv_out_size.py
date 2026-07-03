import torch
import numpy as np


def get_conv_out_size(conv_block, input_size):
    test = torch.zeros(1, *input_size)
    x = conv_block(test)
    out_size = int(np.prod(x.size()))
    return out_size
