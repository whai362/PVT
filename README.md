# Updates
- (2020/06/21) Code of PVTv2 is released! PVTv2 largely improves PVT and works better than Swin Transformer with ImageNet-1K pre-training. Paper will be released recently.

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

Detection configs & models see >>>[here](detection/)<<<.

- PVTv1 on COCO

|    Method   | Lr schd | box AP | mask AP | 
| :-------------: | :-----: | :-----: | :------:
|    PVT-Tiny + RetinaNet | 1x | 36.7    | - |
|    PVT-Small + RetinaNet | 1x | 40.4    | - |
|    PVT-Tiny + Mask RCNN  | 1x | 36.7    | 35.1 |
|    PVT-Small + Mask RCNN  | 1x | 40.4    | 37.8 |
|    PVT-Small + DETR  | 50ep | 34.7    | - |

- PVTv2 on COCO

### Semantic Segmentation

- PVTv1 on ADE20K

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
