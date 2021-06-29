import argparse

import torch
from mmcv import Config, DictAction

from mmdet.models import build_detector

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
import pvt
import pvt_v2
from mmcv.cnn.utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


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


def lisra_flops(h, w, dim):
    return 2 * h * w * 7 * 7 * dim


def get_vit_flops(net, input_shape, patch_size):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)
    _, H, W = input_shape
    stage = mha_flops(H // patch_size, W // patch_size,
                      net.blocks[0].attn.dim,
                      net.blocks[0].attn.num_heads) * len(net.blocks)
    flops += stage
    return flops_to_string(flops), params_to_string(params)


def get_pvt_flops(net, input_shape):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)

    net = net.backbone
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
    flops += stage1 + stage2 + stage3 + stage4
    return flops_to_string(flops), params_to_string(params)


def get_pvt_li_flops(net, input_shape):
    flops, params = get_model_complexity_info(net, input_shape, as_strings=False)

    net = net.backbone
    _, H, W = input_shape
    stage1 = lisra_flops(H // 4, W // 4,
                         net.block1[0].attn.dim) * len(net.block1)
    stage2 = lisra_flops(H // 8, W // 8,
                         net.block2[0].attn.dim) * len(net.block2)
    stage3 = lisra_flops(H // 16, W // 16,
                         net.block3[0].attn.dim) * len(net.block3)
    stage4 = lisra_flops(H // 32, W // 32,
                         net.block4[0].attn.dim) * len(net.block4)
    flops += stage1 + stage2 + stage3 + stage4
    return flops_to_string(flops), params_to_string(params)


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
                format(model.__class__.__name__))
    # calculate CNN-based models' flops
    # flops, params = get_model_complexity_info(model, input_shape)

    # calculate pvtv1/pvtv2 flops
    # flops, params = get_pvt_flops(model, input_shape)

    # calculate pvtv2-li flops
    flops, params = get_pvt_li_flops(model, input_shape)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
