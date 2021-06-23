# Updates
- (21/06/2020) Code of PVTv2 is released! PVTv2 largely improves PVTv1 and also being better than Swin Transformer on classification/object detection with ImageNet-1K pre-training. Paper will be released recently.

# Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
This repository is the official implementation of PVT in classification, object detection, and semantic segmentation tasks.


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

## Model Zoo

### Image Classification

Classification configs & models see >>>[here](classification/)<<<.

- PVTv1 on ImageNet-1K

| name | Size | acc@1 | #params (M) | Config | url |
| --- | --- | --- | --- | --- | --- |
| PVT-Tiny | 224 | 75.1 | 13.2 | [config](classification/configs/pvt/pvt_tiny.py) | [51M](https://drive.google.com/file/d/1yau8uMRl-mnlTAUn4I7vypss3wjVltt5/view?usp=sharing) |
| PVT-Small | 224 | 79.8 | 24.5 | [config](classification/configs/pvt/pvt_small.py) |[93M](https://drive.google.com/file/d/1ds9Rb9wRh9IzGV0CZMM0hnS0QAM_qyIF/view?usp=sharing) |
| PVT-Medium | 224 | 81.2 | 44.2 | [config](classification/configs/pvt/pvt_medium.py) |[168M](https://drive.google.com/file/d/1c2EkzszygPET83h-w4eh-Ef4V_d1a8kw/view?usp=sharing) |
| PVT-Large | 224 | 81.7 | 61.4 | [config](classification/configs/pvt/pvt_large.py) | [234M](https://drive.google.com/file/d/1C07_swTQeWvppIzQrl_0H7UDk4SsalkJ/view?usp=sharing) |

- PVTv2 on ImageNet-1K

| name | Size | acc@1 | #params (M) | Config | url |
| --- | --- | --- | --- | --- | --- |
| PVT-V2-B0 | 224 | 70.5 | 3.4 | [config](configs/pvt_v2/pvt_v2_b0.py) | [14M](https://drive.google.com/file/d/1qnqChpm93vtXULeTuCT_0mJ2ZKIDc-Qo/view?usp=sharing) |
| PVT-V2-B1 | 224 | 78.7 | 13.1 | [config](configs/pvt_v2/pvt_v2_b1.py) | [54M](https://drive.google.com/file/d/1aM0KFE3f-qIpP3xfhihlULF0-NNuk1m7/view?usp=sharing) |
| PVT-V2-B2-Linear | 224 | 82.1 | 22.6 | [config](configs/pvt_v2/pvt_v2_b2_li.py) | [86M](https://drive.google.com/file/d/1_HOJJCIGMMg6RztYAgzbTUge0m28rkZw/view?usp=sharing) |
| PVT-V2-B2 | 224 | 82.0 | 25.4 | [config](configs/pvt_v2/pvt_v2_b2.py) | [97M](https://drive.google.com/file/d/1snw4TYUCD5z4d3aaId1iBdw-yUKjRmPC/view?usp=sharing) |
| PVT-V2-B3 | 224 | 83.1 | 45.2 | [config](configs/pvt_v2/pvt_v2_b3.py) | [173M](https://drive.google.com/file/d/1PzTobv3pu5R3nb3V3lF6_DVnRDBtSmmS/view?usp=sharing) |
| PVT-V2-B4 | 224 | 83.6 | 62.6 | [config](configs/pvt_v2/pvt_v2_b4.py) | [239M](https://drive.google.com/file/d/1LW-0CFHulqeIxV2cai45t-FyLNKGc5l0/view?usp=sharing) |
| PVT-V2-B5 | 224 | 83.8 | 82.0 | [config](configs/pvt_v2/pvt_v2_b5.py) | [313M](https://drive.google.com/file/d/1TKQIdpOFoFs9H6aApUNJKDUK95l_gWy0/view?usp=sharing) |

### Object Detection 

Detection configs & models see >>>[here](detection/)<<<.

- PVTv1 on COCO

|    Method   | Lr schd | box AP | mask AP | Config | Download  |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: |
|    PVT-Tiny + RetinaNet | 1x | 36.7    | - | [config](detection/configs/retinanet_pvt_t_fpn_1x_coco.py)  | Todo. | |
|    PVT-Small + RetinaNet | 1x | 40.4    | - | [config](detection/configs/retinanet_pvt_s_fpn_1x_coco.py)  | [model](https://drive.google.com/file/d/1U02ngyT_IYxS8SlU3WXf5r0TFsoBE3Lm/view?usp=sharing) |
|    PVT-Tiny + Mask RCNN  | 1x | 36.7    | 35.1 | [config](detection/configs/mask_rcnn_pvt_t_fpn_1x_coco.py)  | Todo. |
|    PVT-Small + Mask RCNN  | 1x | 40.4    | 37.8 | [config](detection/configs/mask_rcnn_pvt_s_fpn_1x_coco.py)  | Todo. |
|    PVT-Small + DETR  | 50ep | 34.7    | - | [config](detection/configs/detr_pvt_s_8x2_50ep_coco.py)  | Todo. |

- PVTv2 on COCO

### Semantic Segmentation

- PVTv1 on ADE20K

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
