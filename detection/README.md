# Applying PVT to Object Detection

Our detection code is developed on top of [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).

For details see [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/pdf/2102.12122.pdf). 

If you use this code for a paper please cite:

PVTv1
```
@misc{wang2021pyramid,
      title={Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions}, 
      author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
      year={2021},
      eprint={2102.12122},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

PVTv2
```
@misc{wang2021pvtv2,
      title={PVTv2: Improved Baselines with Pyramid Vision Transformer}, 
      author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
      year={2021},
      eprint={2106.13797},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Usage

Install [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).

or

```
pip install mmdet==2.13.0 --user
```

Apex (optional):
```
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install --cpp_ext --cuda_ext --user
```

If you would like to disable apex, modify the type of runner as `EpochBasedRunner` and comment out the following code block in the configuration files:
```
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
```

## Data preparation

Prepare COCO according to the guidelines in [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).


## Results and models

- PVTv2 on COCO


| Method     | Backbone | Pretrain    | Lr schd | Aug | box AP | mask AP | Config                                               | Download |
|------------|----------|-------------|:-------:|:---:|:------:|:-------:|------------------------------------------------------|----------|
| RetinaNet  | PVTv2-b0 | ImageNet-1K |    1x   |  No |  37.2  |    -    | [config](configs/retinanet_pvt_v2_b0_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1190iSH3oas_71DPEVjMK9JTn59RYdF3T/view?usp=sharing) & [model](https://drive.google.com/file/d/1K6OkU3CYVglnLSDSvsY8HpDcISB6eKzM/view?usp=sharing) |
| RetinaNet  | PVTv2-b1 | ImageNet-1K |    1x   |  No |  41.2  |    -    | [config](configs/retinanet_pvt_v2_b1_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/19Wsg25yvKdiqMjXWIFEe-DcaHlkw3iSv/view?usp=sharing) & [model](https://drive.google.com/file/d/1UyBfxAyQygVgAtBeynXrG2iCJj70kiP9/view?usp=sharing) |
| RetinaNet  | PVTv2-b2-li | ImageNet-1K |    1x   |  No |  43.6  |    -    | [config](configs/retinanet_pvt_v2_b3_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1PRSG3q0M_ZztMMbTxB_961ta4I626T-T/view?usp=sharing) & [model](https://drive.google.com/file/d/1v3j4D1FZuasPi6lGHoHM3bok7PM8F1sg/view?usp=sharing) |
| RetinaNet  | PVTv2-b2 | ImageNet-1K |    1x   |  No |  44.6  |    -    | [config](configs/retinanet_pvt_v2_b2_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1AMkXwopXLJtW71zT3MjXo0YdoojUbxpQ/view?usp=sharing) & [model](https://drive.google.com/file/d/1VqrLiQ0329HpqiG3BU3q0LoXi6ncS1_k/view?usp=sharing) |
| RetinaNet  | PVTv2-b3 | ImageNet-1K |    1x   |  No |  45.9  |    -    | [config](configs/retinanet_pvt_v2_b3_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1L59JWC2jepRMT-l5lo8bqygSUGixfGsr/view?usp=sharing) & [model](https://drive.google.com/file/d/1Lz4qRtDoYT8RvDpVxJCvstM3qTtHlLqL/view?usp=sharing) |
| RetinaNet  | PVTv2-b4 | ImageNet-1K |    1x   |  No |  46.1  |    -    | [config](configs/retinanet_pvt_v2_b4_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1uzFo1W0ARXZfIxBOAGDRi56LpUVIXQNu/view?usp=sharing) & [model](https://drive.google.com/file/d/1GCiE6tniZrG36vumnGi9d79xQEXCS-l2/view?usp=sharing) |
| RetinaNet  | PVTv2-b5 | ImageNet-1K |    1x   |  No |  46.2  |    -    | [config](configs/retinanet_pvt_v2_b5_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/10bxZEXFQSTVWOWjx2WOgKpoohJS51iPd/view?usp=sharing) & [model](https://drive.google.com/file/d/10cUAXpajabSpAJVSPRNywkOiUlgIjN0e/view?usp=sharing) |
| Mask R-CNN | PVTv2-b0 | ImageNet-1K |    1x   |  No |  38.2  |   36.2  | [config](configs/mask_rcnn_pvt_v2_b0_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1F4ILBaFnsjwB-_3H8R9lAuf6aZXf8i-V/view?usp=sharing) & [model](https://drive.google.com/file/d/1eRDCU0Erv-kWwCFCwU_VljdjpHk9ktAY/view?usp=sharing) |
| Mask R-CNN | PVTv2-b1 | ImageNet-1K |    1x   |  No |  41.8  |   38.8  | [config](configs/mask_rcnn_pvt_v2_b1_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1huXFBdZjAtW2PjRUGq5ByNFiuNvy79Va/view?usp=sharing) & [model](https://drive.google.com/file/d/1Y4xgILkl7bh3-DV3rksZTGlN_VTLAIkO/view?usp=sharing) |
| Mask R-CNN | PVTv2-b2-li | ImageNet-1K |    1x   |  No |  44.1  |   40.5  | [config](configs/mask_rcnn_pvt_v2_b3_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1_SsUAOrH73OpSkhx0g15i6bxCn9I826Y/view?usp=sharing) & [model](https://drive.google.com/file/d/1DWUryElNqWzaNuPafWL0GuOSQxqvqxba/view?usp=sharing) |
| Mask R-CNN | PVTv2-b2 | ImageNet-1K |    1x   |  No |  45.3  |   41.2  | [config](configs/mask_rcnn_pvt_v2_b2_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1s0L3Dfk7GIKpSmIX0zFRAArADaJy9MeM/view?usp=sharing) & [model](https://drive.google.com/file/d/1F3-FBjDLkskZFvhECO3sSkHwBg8Mx_j0/view?usp=sharing) |
| Mask R-CNN | PVTv2-b3 | ImageNet-1K |    1x   |  No |  47.0  |   42.5  | [config](configs/mask_rcnn_pvt_v2_b3_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/17fDDXDVim6rFKcrDZCp8ox0_RbeN4yPd/view?usp=sharing) & [model](https://drive.google.com/file/d/1Uq9KpUSLt1-B_6tgVTc1-neuxYBMTI1S/view?usp=sharing) |
| Mask R-CNN | PVTv2-b4 | ImageNet-1K |    1x   |  No |  47.5  |   42.7  | [config](configs/mask_rcnn_pvt_v2_b4_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/1FVWU1ohn19DBOuCZPb8A9DFfrMa5yYm2/view?usp=sharing) & [model](https://drive.google.com/file/d/1IpdgEHAe3XNlIldk6drzOWRcE-wFCv7v/view?usp=sharing) |
| Mask R-CNN | PVTv2-b5 | ImageNet-1K |    1x   |  No |  47.4  |   42.5  | [config](configs/mask_rcnn_pvt_v2_b5_fpn_1x_coco.py) | [log](https://drive.google.com/file/d/19LN-8TWsrVKsF5aBzXiqKva5mQrAusDw/view?usp=sharing) & [model](https://drive.google.com/file/d/1BvI5XXaGbv3tbLrXbVQ5K45gFVEHbBGX/view?usp=sharing) |


| Method     | Backbone | Pretrain    | Lr schd | Aug | box AP | mask AP | Config                                               | Download |
|------------|----------|-------------|:-------:|:---:|:------:|:-------:|------------------------------------------------------|----------|
| Mask R-CNN | PVTv2-b0 | ImageNet-1K |    3x   |  Yes |  41.6  |   38.2  | config | log & model |
| Mask R-CNN | PVTv2-b2 | ImageNet-1K |    3x   |  Yes |  47.8  |   43.1  | config | log & model |


| Method        | Backbone        | Pretrain    | Lr schd | Aug | box AP | mask AP | Config     | Download |
|---------------|-----------------|-------------|:-------:|:---:|:------:|:-------:|------------|----------|
| Cascade Mask R-CNN | PVTv2-b2-Linear | ImageNet-1K |    3x   | Yes |  50.9  |    44.0    | [config](configs/cascade_mask_rcnn_pvt_v2_b2_li_fpn_3x_mstrain_fp16.py) | [log](https://drive.google.com/file/d/1X_DC4yd89t4MJjQt9XmuCwx1hRmklN3z/view?usp=sharing) & [model](https://drive.google.com/file/d/1dG4O-M0EqKYdTtZqJdRjJopiwwnoakee/view?usp=sharing)    |
| Cascade Mask R-CNN | PVTv2-b2        | ImageNet-1K |    3x   | Yes |  51.1  |    44.4    | [config](configs/cascade_mask_rcnn_pvt_v2_b2_fpn_3x_mstrain_fp16.py) | [log](https://drive.google.com/file/d/1gKEa_lUvm3Okonk33wgUjzfzVijEch-2/view?usp=sharing) & [model](https://drive.google.com/file/d/11jmqwLQSqQ1zin9D2sRYqeC8YfzGaN3V/view?usp=sharing) |
| ATSS          | PVTv2-b2-Linear | ImageNet-1K |    3x   | Yes |  48.9  |    -    | [config](configs/atss_pvt_v2_b2_li_fpn_3x_mstrain_fp16.py) | [log](https://drive.google.com/file/d/1pg2O6gC5zKvFnAuC98wsexcx7mAqmv16/view?usp=sharing) & [model](https://drive.google.com/file/d/1CB4teTBwOpofCrHM91QvKKFLKcVMzZVi/view?usp=sharing)   |
| ATSS          | PVTv2-b2        | ImageNet-1K |    3x   | Yes |  49.9  |    -    | [config](configs/atss_pvt_v2_b2_fpn_3x_mstrain_fp16.py) | [log](https://drive.google.com/file/d/1Vnf8-BszhTEkOQqwLA2-XLeuMT9n5ceR/view?usp=sharing) & [model](https://drive.google.com/file/d/1TKbj-i7oLgC7zstFuV0Neumu4iKMpBGh/view?usp=sharing)    |
| GFL           | PVTv2-b2-Linear | ImageNet-1K |    3x   | Yes |  49.2  |    -    | [config](configs/gfl_pvt_v2_b2_li_fpn_3x_mstrain_fp16.py) | [log](https://drive.google.com/file/d/1hqieuwCe79HAVMMVz8sEZsnG-R74Z_AO/view?usp=sharing) & [model](https://drive.google.com/file/d/1CnXlOEs9g7-LAoaDFcukTh5x0R4popZp/view?usp=sharing)    |
| GFL           | PVTv2-b2        | ImageNet-1K |    3x   | Yes |  50.2  |    -    | [config](configs/gfl_pvt_v2_b2_fpn_3x_mstrain_fp16.py) | [log](https://drive.google.com/file/d/1AEMecyBnsomn4bxj1ySMxFdCsi8KCzGT/view?usp=sharing) & [model](https://drive.google.com/file/d/1XODtTQ3UAQz75vqhXBddqn7JpQke0vn6/view?usp=sharing) |
| Sparse R-CNN  | PVTv2-b2-Linear | ImageNet-1K |    3x   | Yes |  48.9  |    -    | [config](configs/sparse_rcnn_pvt_v2_b2_li_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py) | [log](https://drive.google.com/file/d/1uVHEwr5FDqlL3UvstpncCuaClU54lig6/view?usp=sharing) & [model](https://drive.google.com/file/d/1W8Wt2WbyhEi0JOUblaEcH9gx0I6z1wAv/view?usp=sharing) |
| Sparse R-CNN  | PVTv2-b2        | ImageNet-1K |    3x   | Yes |  50.1  |    -    | [config](configs/sparse_rcnn_pvt_v2_b2_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py) | [log](https://drive.google.com/file/d/1hDJwwICMmFqqF0A2Z35uNR6C5nI-2m22/view?usp=sharing) & [model](https://drive.google.com/file/d/1xtn-wD_nYSwudF1SqsSHl7opEXM4dhPN/view?usp=sharing)    |

- PVTv1 on COCO

| Method           | Backbone  | Pretrain    | Lr schd | box AP | mask AP | Config                                               | Download                                                                                    |
|------------------|-----------|-------------|:-------:|:------:|:-------:|------------------------------------------------------|---------------------------------------------------------------------------------------------|
| RetinaNet        | PVT-Tiny  | ImageNet-1K |    1x   |  36.7  |    -    | [config](configs/retinanet_pvt_t_fpn_1x_coco.py)     | [log](https://drive.google.com/file/d/1w5giZkGZ0raFnl6TE8V7G3vbFoeRDqWC/view?usp=sharing) & [model](https://drive.google.com/file/d/1ZbS-g3oqAChkYiDYTiZLzw61lal2pgbl/view?usp=sharing) |
| RetinaNet (640x) | PVT-Small | ImageNet-1K |    1x   |  38.7  |    -    | [config](configs/retinanet_pvt_s_fpn_1x_coco_640.py) | [log](https://drive.google.com/file/d/1lkDW5jmUz9YwED4BNiUYFSi9bkcAB36y/view?usp=sharing) & [model](https://drive.google.com/file/d/1L5wh2rYsVnuC_CEeFE6yMhU1kENt2gnk/view?usp=sharing) |
| RetinaNet (800x) | PVT-Small | ImageNet-1K |    1x   |  40.4  |    -    | [config](configs/retinanet_pvt_s_fpn_1x_coco.py)     | log & [model](https://drive.google.com/file/d/1U02ngyT_IYxS8SlU3WXf5r0TFsoBE3Lm/view?usp=sharing) |
| RetinaNet        | PVT-Medium  | ImageNet-1K |    1x   |  41.9  |    -    | [config](configs/retinanet_pvt_m_fpn_1x_coco.py)     | [log](https://drive.google.com/file/d/1sbHtmaFoQvbjE881MqmgOw1QtwrD2MVF/view?usp=sharing) & [model](https://drive.google.com/file/d/1DJCmbrpLJhQi20ztSCmtruDaXtqOsrns/view?usp=sharing) |
| RetinaNet        | PVT-Large  | ImageNet-1K |    1x   |  42.6  |    -    | [config](configs/retinanet_pvt_l_fpn_1x_coco.py)     | [log](https://drive.google.com/file/d/1PjvUY1tRZ7Qhdtx1MQ9J0lPMRZgV3x1n/view?usp=sharing) & [model](https://drive.google.com/file/d/1zG_Ji_GamjQgPbe6hHOjZlIHs5jZbVFL/view?usp=sharing) |
| Mask R-CNN        | PVT-Tiny  | ImageNet-1K |    1x   |  36.7  |   35.1  | [config](configs/mask_rcnn_pvt_t_fpn_1x_coco.py)     | [log](https://drive.google.com/file/d/1PE__Sp2tgKIYkJaUa0V8q-GSxyLkcWke/view?usp=sharing) & [model](https://drive.google.com/file/d/1JGcl7ZnDIf-qQjrCXVtb71XeRWOIu7Xf/view?usp=sharing) |
| Mask R-CNN        | PVT-Small | ImageNet-1K |    1x   |  40.4  |   37.8  | [config](configs/mask_rcnn_pvt_s_fpn_1x_coco.py)     | [log](https://drive.google.com/file/d/1qrP563Tv_Xc-gMDlX6cXiyhLpM0Yi_iN/view?usp=sharing) & [model](https://drive.google.com/file/d/1XGJQTivM_KviRDVvKTPgkJQhFco2oXSA/view?usp=sharing) |
| Mask R-CNN        | PVT-Medium | ImageNet-1K |    1x   |  42.0  |   39.0  | [config](configs/mask_rcnn_pvt_m_fpn_1x_coco.py)     | [log](https://drive.google.com/file/d/1773WWorAkeh5mVvvbJoqewAB4W0qa8yb/view?usp=sharing) & [model](https://drive.google.com/file/d/1m0TFlJOVSLaAA4alVqp_VRckg0uRGTJ-/view?usp=sharing) |
| Mask R-CNN        | PVT-Large | ImageNet-1K |    1x   |  42.9  |   39.5  | [config](configs/mask_rcnn_pvt_l_fpn_1x_coco.py)     | [log](https://drive.google.com/file/d/1PjvUY1tRZ7Qhdtx1MQ9J0lPMRZgV3x1n/view?usp=sharing) & [model](https://drive.google.com/file/d/1zG_Ji_GamjQgPbe6hHOjZlIHs5jZbVFL/view?usp=sharing) |
| DETR             | PVT-Small | ImageNet-1K |   50ep  |  34.7  |    -    | [config](configs/detr_pvt_s_8x2_50ep_coco.py)        | [log](https://drive.google.com/file/d/1svBJbh8sHK2Ea0ORY5g2PxVr_O16AI0i/view?usp=sharing) & [model](https://drive.google.com/file/d/1rIf2qUEvaeV2ZMzhVyiXxEplE0bCi3Jv/view?usp=sharing) |


## Evaluation
To evaluate PVT-Small + RetinaNet (640x) on COCO val2017 on a single node with 8 gpus run:
```
dist_test.sh configs/retinanet_pvt_s_fpn_1x_coco_640.py /path/to/checkpoint_file 8 --out results.pkl --eval bbox
```
This should give
```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.593
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.408
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.212
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.416
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.544
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.545
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.545
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.545
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.329
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.583
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.721
```

## Training
To train PVT-Small + RetinaNet (640x) on COCO train2017 on a single node with 8 gpus for 12 epochs run:

```
dist_train.sh configs/retinanet_pvt_s_fpn_1x_coco_640.py 8
```

## Demo
```
python demo.py demo.jpg /path/to/config_file /path/to/checkpoint_file
```


## Calculating FLOPS & Params

```
python get_flops.py configs/gfl_pvt_v2_b2_fpn_3x_mstrain_fp16.py
```
This should give
```
Input shape: (3, 1280, 800)
Flops: 260.65 GFLOPs
Params: 33.11 M
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
