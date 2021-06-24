# Updates
- (2020/06/21) Code of PVTv2 is released! PVTv2 largely improves PVTv1 and works better than Swin Transformer with ImageNet-1K pre-training. Paper will be released recently.

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

Classification configs & weights see >>>[here](classification/)<<<.

- PVTv1 on ImageNet-1K

| Method | Size | Acc@1 | #Params (M) |
| --- | --- | --- | --- |
| PVT-Tiny | 224 | 75.1 | 13.2 |
| PVT-Small | 224 | 79.8 | 24.5 |
| PVT-Medium | 224 | 81.2 | 44.2 |
| PVT-Large | 224 | 81.7 | 61.4 |

- PVTv2 on ImageNet-1K

| Method | Size | Acc@1 | #Params (M) |
| --- | --- | --- | --- |
| PVT-V2-B0 | 224 | 70.5 | 3.4 |
| PVT-V2-B1 | 224 | 78.7 | 13.1 |
| PVT-V2-B2-Linear | 224 | 82.1 | 22.6 |
| PVT-V2-B2 | 224 | 82.0 | 25.4 |
| PVT-V2-B3 | 224 | 83.1 | 45.2 |
| PVT-V2-B4 | 224 | 83.6 | 62.6 |
| PVT-V2-B5 | 224 | 83.8 | 82.0 |

### Object Detection 

Detection configs & weights see >>>[here](detection/)<<<.

- PVTv1 on COCO

| Detector  | BackBone  | Pretrain    | Lr schd | box AP | mask AP |
|-----------|-----------|-------------|:-------:|:------:|:-------:|
| RetinaNet | PVT-Tiny  | ImageNet-1K |    1x   |  36.7  |    -    |
| RetinaNet | PVT-Small | ImageNet-1K |    1x   |  40.4  |    -    |
| Mask RCNN | PVT-Tiny  | ImageNet-1K |    1x   |  36.7  |   35.1  |
| Mask RCNN | PVT-Small | ImageNet-1K |    1x   |  40.4  |   37.8  |
| DETR      | PVT-Small | ImageNet-1K |   50ep  |  34.7  |    -    |

- PVTv2 on COCO


|   Method   | Backbone | Pretrain    | Lr schd | Aug | box AP | mask AP |
|:----------:|----------|-------------|:-------:|:---:|:------:|:-------:|
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


| Method        | BackBone        | Lr schd | Aug | box AP |
|---------------|-----------------|:-------:|:---:|:------:|
| Cascade R-CNN | PVTv2-b2-Linear |    3x   | Yes |  50.9  |
| Cascade R-CNN | PVTv2-b2        |    3x   | Yes |  51.1  |
| ATSS          | PVTv2-b2-Linear |    3x   | Yes |  48.9  |
| ATSS          | PVTv2-b2        |    3x   | Yes |  49.9  |
| GFL           | PVTv2-b2-Linear |    3x   | Yes |  49.2  |
| GFL           | PVTv2-b2        |    3x   | Yes |  50.2  |
| Sparse R-CNN  | PVTv2-b2-Linear |    3x   | Yes |  48.9  |
| Sparse R-CNN  | PVTv2-b2        |    3x   | Yes |  50.1  |

- PVTv2 on COCO

### Semantic Segmentation

- PVTv1 on ADE20K

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
