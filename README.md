# Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions

This repository contains PyTorch evaluation code, training code and pretrained models for PVT (Pyramid Vision Transformer).

Like ResNet, PVT is a pure transformer backbone that can be easily plugged in most downstream task models.

With a comparable number of parameters, PVT-Small+RetinaNet achieves 40.4 AP on the COCO dataset, surpassing ResNet50+RetinNet (36.3 AP) by 4.1 AP.

<div align="center">
  <img src="https://github.com/whai362/PVT/blob/main/.github/pvt.png">
</div>
<p align="center">
  Figure 1: Performance of RetinaNet 1x with different backbones.
</p>

This repository is developed on top of [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and [deit](https://github.com/facebookresearch/deit).

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
- PVT + Semantic FPN configs & models
- PVT + DETR/[Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN) config & models
- PVT + Trans2Seg config & models

## Usage

First, clone the repository locally:
```
git clone https://github.com/whai362/PVT.git
```
Then, install PyTorch 1.6.0+ and torchvision 0.7.0+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Model Zoo

### Object Detection 

Detection configs & models see [here](https://github.com/whai362/PVT/tree/main/detection).

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


### Image Classification

We provide baseline PVT models pretrained on ImageNet 2012.

| name | acc@1 | #params (M) | url |
| --- | --- | --- | --- |
| PVT-Tiny | 75.1 | 13.2 | [51 M](https://drive.google.com/file/d/1NLw3hRJMoOQbUXAoftg8tUFCWuTwUIQz/view?usp=sharing), [PyTorch<=1.5](https://drive.google.com/file/d/1yau8uMRl-mnlTAUn4I7vypss3wjVltt5/view?usp=sharing) |
| PVT-Small | 79.8 | 24.5 | [93 M](https://drive.google.com/file/d/1vtcyoU8KUqNzktlMGXZrYcMRsNNiVZFQ/view?usp=sharing), [PyTorch<=1.5](https://drive.google.com/file/d/1ds9Rb9wRh9IzGV0CZMM0hnS0QAM_qyIF/view?usp=sharing) |
| PVT-Medium | 81.2 | 44.2 | [168M](https://drive.google.com/file/d/1c2EkzszygPET83h-w4eh-Ef4V_d1a8kw/view?usp=sharing) |
| PVT-Large | 81.7 | 61.4 | [234M](https://drive.google.com/file/d/1C07_swTQeWvppIzQrl_0H7UDk4SsalkJ/view?usp=sharing) |

## Evaluation
To evaluate a pre-trained PVT-Small on ImageNet val with a single GPU run:
```
sh dist_train.sh pvt_small 1 /path/to/checkpoint_root --data-path /path/to/imagenet --resume /path/to/checkpoint_file --eval
```
This should give
```
* Acc@1 79.764 Acc@5 94.950 loss 0.885
Accuracy of the network on the 50000 test images: 79.8%
```

## Training
To train PVT-Small on ImageNet on a single node with 8 gpus for 300 epochs run:

```
sh dist_train.sh pvt_small 8 /path/to/checkpoint_root --data-path /path/to/imagenet
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
