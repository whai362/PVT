import torchvision.models as models
import pvt
import torch
from mmcv.cnn.utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string


def mha_flops(h, w, dim, num_heads):
    dim_h = dim / num_heads
    n = h * w
    f1 = n * dim_h * n * num_heads
    f2 = n * n * dim_h * num_heads
    return f1 + f2


def sra_flops(h, w, r, dim, num_heads):
    dim_h = dim / num_heads
    n1 = h * w
    n2 = h / r * w / r

    f1 = n1 * dim_h * n2 * num_heads
    f2 = n1 * n2 * dim_h * num_heads

    return f1 + f2


def get_pvt_flops(net, input_shape):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    _, H, W = input_shape
    stage1 = sra_flops(H // 4, W // 4,
                       net.block1[0].attn.sr_ratio,
                       net.block1[0].attn.dim,
                       net.block1[0].attn.num_heads) * len(net.block1)
    stage2 = sra_flops(H // 8, W // 8,
                       net.block2[0].attn.sr_ratio,
                       net.block2[0].attn.dim,
                       net.block2[0].attn.num_heads) * len(net.block2)
    stage3 = sra_flops(H // 16, W // 16,
                       net.block3[0].attn.sr_ratio,
                       net.block3[0].attn.dim,
                       net.block3[0].attn.num_heads) * len(net.block3)
    stage4 = sra_flops(H // 32, W // 32,
                       net.block4[0].attn.sr_ratio,
                       net.block4[0].attn.dim,
                       net.block4[0].attn.num_heads) * len(net.block4)
    print(stage1 + stage2 + stage3 + stage4)
    flops += stage1 + stage2 + stage3 + stage4
    return flops_to_string(flops), params_to_string(params)


def get_vit_flops(net, input_shape, patch_size):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    _, H, W = input_shape
    stage = mha_flops(H // patch_size, W // patch_size,
                      net.blocks[0].attn.dim,
                      net.blocks[0].attn.num_heads) * len(net.blocks)
    flops += stage
    return flops_to_string(flops), params_to_string(params)


with torch.cuda.device(0):
    # net = models.vit_small_patch16_224()
    # input_shape = (3, 224, 224)
    # flops, params = get_vit_flops(net, input_shape, 16)
    # print(flops)
    net = pvt.pvt_small()
    input_shape = (3, 224, 224)
    flops, params = get_pvt_flops(net, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
