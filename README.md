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

This repository is developed on the top of [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and [deit](https://github.com/facebookresearch/deit).

For details see [Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions](None). 

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

# Todo List
- ImageNet model weights
- PVT + RetinaNet/Mask R-CNN config & models
- PVT + Semantic FPN config & models
- PVT + DETR config & models
- PVT + Trans2Seg config & models

# Model Zoo

We provide baseline PVT models pretrained on ImageNet 2012.

| name | acc@1 | #params (M) | url |
| --- | --- | --- | --- |
| PVT-Tiny | 75.1 | 13.2 | Todo. |
| PVT-Small | 79.8 | 24.5 | Todo. |
| PVT-Medium | 81.2 | 44.2 | Todo. |
| PVT-Large | 81.7 | 61.4 | Todo. |

Before using it, make sure you have the pytorch-image-models package [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models) by [Ross Wightman](https://github.com/rwightman) installed. Note that our work relies of the augmentations proposed in this library.

# Usage

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

## Evaluation
To evaluate a pre-trained PVT-Small on ImageNet val with a single GPU run:
```
sh dist_train.sh pvt_small 1 /path/to/checkpoint_root --data-path /path/to/imagenet --resume /path/to/checkpoint_file --eval
```
This should give
```
Todo.
```

## Training
To train PVT-Small on ImageNet on a single node with 8 gpus for 300 epochs run:

```
sh dist_train.sh pvt_small 8 /path/to/checkpoint_root --data-path /path/to/imagenet
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
