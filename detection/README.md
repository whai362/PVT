# Applying PVT to Object Detection

Our detection code is developed on top of [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).

For details see [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](https://arxiv.org/pdf/2102.12122.pdf). 

If you use this code for a paper please cite:

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

## Todo List
- PVT-Medium/-Large + RetinaNet/Mask R-CNN
- PVTv2-b0/b1/b2/b3/b4/b5 + RetinaNet/Mask R-CNN
- PVTv2-b2 + Sparse R-CNN/Cascade R-CNN/GFL/ATSS


## Usage

Install [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).


## Data preparation

First, prepare COCO according to the guidelines in [MMDetection v2.13.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).

Then, download the [weights pretrained on ImageNet](../classification/README.md), and put them in a folder `pretrained/`

## Results and models

- PVTv1 on COCO

|    Method   | Lr schd | box AP | mask AP | Config | Download  |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: |
|    PVT-Tiny + RetinaNet  | 1x | 36.7    | - | [config](configs/retinanet_pvt_t_fpn_1x_coco.py)  | Todo. |
|    PVT-Small + RetinaNet (640x)  | 1x | 38.7    | - |  [config](configs/retinanet_pvt_s_fpn_1x_coco_640.py)  | [model](https://drive.google.com/file/d/1L5wh2rYsVnuC_CEeFE6yMhU1kENt2gnk/view?usp=sharing) |
|    PVT-Small + RetinaNet (800x)  | 1x | 40.4    | - | [config](configs/retinanet_pvt_s_fpn_1x_coco.py)  | [model](https://drive.google.com/file/d/1U02ngyT_IYxS8SlU3WXf5r0TFsoBE3Lm/view?usp=sharing) |
|    PVT-Tiny + Mask RCNN  | 1x | 36.7    | 35.1 | [config](configs/mask_rcnn_pvt_t_fpn_1x_coco.py)  | Todo. |
|    PVT-Small + Mask RCNN  | 1x | 40.4    | 37.8 | [config](configs/mask_rcnn_pvt_s_fpn_1x_coco.py)  | Todo. |
|    PVT-Small + DETR  | 50ep | 34.7    | - | [config](configs/detr_pvt_s_8x2_50ep_coco.py)  | Todo. |

- PVTv2 on COCO


|    Method   | Lr schd | box AP | mask AP | Config | Download  |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: |
|    PVTv2-b0 + RetinaNet  | 1x | 37.2    | - | [config](configs/retinanet_pvtv2_b0_fpn_1x_coco.py)  | Todo. |
|    PVTv2-b1 + RetinaNet  | 1x | 41.2    | - | [config](configs/retinanet_pvtv2_b1_fpn_1x_coco.py)  | Todo. |
|    PVTv2-b2 + RetinaNet  | 1x | 44.6    | - | [config](configs/retinanet_pvtv2_b2_fpn_1x_coco.py)  | Todo. |
|    PVTv2-b3 + RetinaNet  | 1x | 45.9    | - | [config](configs/retinanet_pvtv2_b3_fpn_1x_coco.py)  | Todo. |
|    PVTv2-b4 + RetinaNet  | 1x | 46.1    | - | [config](configs/retinanet_pvtv2_b4_fpn_1x_coco.py)  | Todo. |
|    PVTv2-b5 + RetinaNet  | 1x | 46.2    | - | [config](configs/retinanet_pvtv2_b5_fpn_1x_coco.py)  | Todo. |
|    PVTv2-b0 + Mask R-CNN  | 1x | 38.2    | 36.2 | [config](configs/mask_rcnn_pvtv2_b0_fpn_1x_coco.py)  | Todo. |
|    PVTv2-b1 + Mask R-CNN  | 1x | 41.8    | 38.8 | [config](configs/mask_rcnn_pvtv2_b1_fpn_1x_coco.py)  | Todo. |
|    PVTv2-b2 + Mask R-CNN  | 1x | 45.3    | 41.2 | [config](configs/mask_rcnn_pvtv2_b2_fpn_1x_coco.py)  | Todo. |
|    PVTv2-b3 + Mask R-CNN  | 1x | 47.0    | 42.5 | [config](configs/mask_rcnn_pvtv2_b3_fpn_1x_coco.py)  | Todo. |
|    PVTv2-b4 + Mask R-CNN  | 1x | 47.5    | 42.7 | [config](configs/mask_rcnn_pvtv2_b4_fpn_1x_coco.py)  | Todo. |
|    PVTv2-b5 + Mask R-CNN  | 1x | 47.4    | 42.5 | [config](configs/mask_rcnn_pvtv2_b5_fpn_1x_coco.py)  | Todo. |

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

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
