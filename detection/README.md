# Applying PVT to Object Detection

Here, we take [MMDetection v2.8.0](https://github.com/open-mmlab/mmdetection/tree/v2.8.0) as an example, applying PVT to RetinaNet and Mask R-CNN.

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


## Usage

Install [MMDetection v2.8.0](https://github.com/open-mmlab/mmdetection/tree/v2.8.0).


## Data preparation

First, prepare COCO according to the guidelines in [MMDetection v2.8.0](https://github.com/open-mmlab/mmdetection/tree/v2.8.0).

Then, download the [weights](https://github.com/whai362/PVT/blob/main/README.md) pretrained on ImageNet, and put them in a folder `pretrained/`

## Results and models

|    Method   | Lr schd | box AP | mask AP | Config | Download  |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: |
|    PVT-Tiny + RetinaNet (800x)  | 1x | 36.7    | - | [config](https://github.com/whai362/PVT/blob/main/detection/configs/retinanet_pvt_t_fpn_1x_coco.py)  | Todo. |
|    PVT-Small + RetinaNet (640x)  | 1x | 38.7    | - |  [config](https://github.com/whai362/PVT/blob/main/detection/configs/retinanet_pvt_s_fpn_1x_coco_640.py)  | [model](https://drive.google.com/file/d/1L5wh2rYsVnuC_CEeFE6yMhU1kENt2gnk/view?usp=sharing) |
|    PVT-Small + RetinaNet (800x)  | 1x | 40.4    | - | [config](https://github.com/whai362/PVT/blob/main/detection/configs/retinanet_pvt_s_fpn_1x_coco.py)  | [model](https://drive.google.com/file/d/1U02ngyT_IYxS8SlU3WXf5r0TFsoBE3Lm/view?usp=sharing) |
|    R50 + DETR  | 50ep | 32.3  | - | [config](https://github.com/whai362/PVT/blob/main/detection/configs/detr_r50_8x2_50ep_coco_baseline.py)  | Todo. |
|    PVT-Small + DETR  | 50ep | 34.7    | - | [config](https://github.com/whai362/PVT/blob/main/detection/configs/detr_pvt_s_8x2_50ep_coco.py)  | Todo. |
|    R50 + DETR  | 50ep | 32.3  | - | [config](https://github.com/whai362/PVT/blob/main/detection/configs/detr_r50_8x2_50ep_coco_baseline.py)  | Todo. |
|    PVT-Tiny + Mask RCNN  | 1x | 36.7    | 35.1 | [config](https://github.com/whai362/PVT/blob/main/detection/configs/mask_rcnn_pvt_t_fpn_1x_coco.py)  | Todo. |
|    PVT-Small + Mask RCNN  | 1x | 40.4    | 37.8 | [config](https://github.com/whai362/PVT/blob/main/detection/configs/mask_rcnn_pvt_s_fpn_1x_coco.py)  | Todo. |

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
