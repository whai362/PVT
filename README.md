# Updates
- (2022/08/09) Application examples for polyp segmentation (polyp-pvt) and vision-language modeling.
- (2020/06/21) Code of PVTv2 is released! PVTv2 largely improves PVTv1 and works better than Swin Transformer with ImageNet-1K pre-training.

# Pyramid Vision Transformer

<div align="center">
  <img width="400", src="./logo.png">
</div>
<p align="center">
  The image is from Transformers: Revenge of the Fallen.
</p>

This repository contains the official implementation of [PVTv1](https://arxiv.org/abs/2102.12122) & [PVTv2](https://arxiv.org/pdf/2106.13797.pdf) in image classification, object detection, and semantic segmentation tasks.


## Model Zoo

### Image Classification

Classification configs & weights see >>>[here](classification/)<<<.

- PVTv2 on ImageNet-1K

| Method           | Size | Acc@1 | #Params (M) |
|------------------|:----:|:-----:|:-----------:|
| PVTv2-B0        |  224 |  70.5 |     3.7     |
| PVTv2-B1        |  224 |  78.7 |     14.0    |
| PVTv2-B2-Linear |  224 |  82.1 |     22.6    |
| PVTv2-B2        |  224 |  82.0 |     25.4    |
| PVTv2-B3        |  224 |  83.1 |     45.2    |
| PVTv2-B4        |  224 |  83.6 |     62.6    |
| PVTv2-B5        |  224 |  83.8 |     82.0    |

- PVTv1 on ImageNet-1K

| Method     | Size | Acc@1 | #Params (M) |
|------------|:----:|:-----:|:-----------:|
| PVT-Tiny   |  224 |  75.1 |     13.2    |
| PVT-Small  |  224 |  79.8 |     24.5    |
| PVT-Medium |  224 |  81.2 |     44.2    |
| PVT-Large  |  224 |  81.7 |     61.4    |


### Object Detection 

Detection configs & weights see >>>[here](detection/)<<<.


- PVTv2 on COCO

#### Baseline Detectors


|   Method   | Backbone | Pretrain    | Lr schd | Aug | box AP | mask AP |
|------------|----------|-------------|:-------:|:---:|:------:|:-------:|
|  RetinaNet | PVTv2-b0 | ImageNet-1K |    1x   |  No |  37.2  |    -    |
|  RetinaNet | PVTv2-b1 | ImageNet-1K |    1x   |  No |  41.2  |    -    |
|  RetinaNet | PVTv2-b2 | ImageNet-1K |    1x   |  No |  44.6  |    -    |
|  RetinaNet | PVTv2-b3 | ImageNet-1K |    1x   |  No |  45.9  |    -    |
|  RetinaNet | PVTv2-b4 | ImageNet-1K |    1x   |  No |  46.1  |    -    |
|  RetinaNet | PVTv2-b5 | ImageNet-1K |    1x   |  No |  46.2  |    -    |
| Mask R-CNN | PVTv2-b0 | ImageNet-1K |    1x   |  No |  38.2  |   36.2  |
| Mask R-CNN | PVTv2-b1 | ImageNet-1K |    1x   |  No |  41.8  |   38.8  |
| Mask R-CNN | PVTv2-b2 | ImageNet-1K |    1x   |  No |  45.3  |   41.2  |
| Mask R-CNN | PVTv2-b3 | ImageNet-1K |    1x   |  No |  47.0  |   42.5  |
| Mask R-CNN | PVTv2-b4 | ImageNet-1K |    1x   |  No |  47.5  |   42.7  |
| Mask R-CNN | PVTv2-b5 | ImageNet-1K |    1x   |  No |  47.4  |   42.5  |


#### Advanced Detectors


| Method             | Backbone        | Pretrain    | Lr schd | Aug | box AP | mask AP |
|--------------------|-----------------|-------------|:-------:|:---:|:------:|:-------:|
| Cascade Mask R-CNN | PVTv2-b2-Linear | ImageNet-1K |    3x   | Yes |  50.9  |   44.0  |
| Cascade Mask R-CNN | PVTv2-b2        | ImageNet-1K |    3x   | Yes |  51.1  |   44.4  |
| ATSS          | PVTv2-b2-Linear | ImageNet-1K |    3x   | Yes |  48.9  |   -   |
| ATSS          | PVTv2-b2        | ImageNet-1K |    3x   | Yes |  49.9  |   -   |
| GFL           | PVTv2-b2-Linear | ImageNet-1K |    3x   | Yes |  49.2  |   -   |
| GFL           | PVTv2-b2        | ImageNet-1K |    3x   | Yes |  50.2  |   -   |
| Sparse R-CNN  | PVTv2-b2-Linear | ImageNet-1K |    3x   | Yes |  48.9  |   -   |
| Sparse R-CNN  | PVTv2-b2        | ImageNet-1K |    3x   | Yes |  50.1  |   -   |

- PVTv1 on COCO

| Detector  | Backbone  | Pretrain    | Lr schd | box AP | mask AP |
|-----------|-----------|-------------|:-------:|:------:|:-------:|
| RetinaNet | PVT-Tiny  | ImageNet-1K |    1x   |  36.7  |    -    |
| RetinaNet | PVT-Small | ImageNet-1K |    1x   |  40.4  |    -    |
| Mask RCNN | PVT-Tiny  | ImageNet-1K |    1x   |  36.7  |   35.1  |
| Mask RCNN | PVT-Small | ImageNet-1K |    1x   |  40.4  |   37.8  |
| DETR      | PVT-Small | ImageNet-1K |   50ep  |  34.7  |    -    |


### Semantic Segmentation

Segmentation configs & weights see >>>[here](segmentation/)<<<.

PVT-v2 + Segmentation see >>>[here](https://github.com/whai362/PVTv2-Seg)<<<.

- PVTv1 on ADE20K

| Method       | Backbone   | Pretrain    | Iters | mIoU |
|--------------|------------|-------------|-------|------|
| Semantic FPN | PVT-Tiny   | ImageNet-1K | 40K   | 35.7 |
| Semantic FPN | PVT-Small  | ImageNet-1K | 40K   | 39.8 |
| Semantic FPN | PVT-Medium | ImageNet-1K | 40K   | 41.6 |
| Semantic FPN | PVT-Large  | ImageNet-1K | 40K   | 42.1 |

### Polyp Segmentation
Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers. [pdf](https://arxiv.org/abs/2108.06932) | [code](https://github.com/DengPingFan/Polyp-PVT)

### Vision-Language Modeling
Masked Vision-Language Transformer in Fashion. [pdf](https://dengpingfan.github.io/papers/[2022][MIR]MVLT.pdf) | [code](https://github.com/GewelsJI/MVLT)

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.


## Citation
If you use this code for a paper, please cite:

PVTv1
```
@inproceedings{wang2021pyramid,
  title={Pyramid vision transformer: A versatile backbone for dense prediction without convolutions},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={568--578},
  year={2021}
}
```

PVTv2
```
@article{wang2021pvtv2,
  title={Pvtv2: Improved baselines with pyramid vision transformer},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Fan, Deng-Ping and Song, Kaitao and Liang, Ding and Lu, Tong and Luo, Ping and Shao, Ling},
  journal={Computational Visual Media},
  volume={8},
  number={3},
  pages={1--10},
  year={2022},
  publisher={Springer}
}
```



## Contact

This repo is currently maintained by Wenhai Wang ([@whai362](https://github.com/whai362)), Enze Xie ([@xieenze](https://github.com/xieenze)), and Zhe Chen ([@czczup](https://github.com/czczup)).
