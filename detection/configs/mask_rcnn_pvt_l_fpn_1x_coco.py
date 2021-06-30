_base_ = [
    '../configs/_base_/models/mask_rcnn_r50_fpn.py',
    '../configs/_base_/datasets/coco_instance.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]
model = dict(
    # pretrained='pretrained/pvt_large.pth',
    pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_large.pth',
    backbone=dict(
        type='pvt_large',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002 / 1.4, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# dataset settings
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1)
