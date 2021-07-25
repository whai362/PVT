# Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions

Our classification code is developed on top of [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) and [deit](https://github.com/facebookresearch/deit).

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


## Todo List
- PVT + ImageNet-22K pre-training.

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

- PVTv2 on ImageNet-1K

| Method           | Size | Acc@1 | #Params (M) | Config                                   | Download                                                                                   |
|------------------|:----:|:-----:|:-----------:|------------------------------------------|--------------------------------------------------------------------------------------------|
| PVT-V2-B0        |  224 |  70.5 |     3.7     | [config](configs/pvt_v2/pvt_v2_b0.py)    | 14M [[Google]](https://drive.google.com/file/d/1qnqChpm93vtXULeTuCT_0mJ2ZKIDc-Qo/view?usp=sharing) [[GitHub]](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth) |
| PVT-V2-B1        |  224 |  78.7 |     14.0    | [config](configs/pvt_v2/pvt_v2_b1.py)    | 54M [[Google]](https://drive.google.com/file/d/1aM0KFE3f-qIpP3xfhihlULF0-NNuk1m7/view?usp=sharing) [[GitHub]](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth) |
| PVT-V2-B2-Linear |  224 |  82.1 |     22.6    | [config](configs/pvt_v2/pvt_v2_b2_li.py) | 86M [[GitHub]](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2_li.pth) |
| PVT-V2-B2        |  224 |  82.0 |     25.4    | [config](configs/pvt_v2/pvt_v2_b2.py)    | 97M [[Google]](https://drive.google.com/file/d/1snw4TYUCD5z4d3aaId1iBdw-yUKjRmPC/view?usp=sharing) [[GitHub]](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth) |
| PVT-V2-B3        |  224 |  83.1 |     45.2    | [config](configs/pvt_v2/pvt_v2_b3.py)    | 173M [[Google]](https://drive.google.com/file/d/1PzTobv3pu5R3nb3V3lF6_DVnRDBtSmmS/view?usp=sharing) [[GitHub]](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth)|
| PVT-V2-B4        |  224 |  83.6 |     62.6    | [config](configs/pvt_v2/pvt_v2_b4.py)    | 239M [[Google]](https://drive.google.com/file/d/1LW-0CFHulqeIxV2cai45t-FyLNKGc5l0/view?usp=sharing) [[GitHub]](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b4.pth)|
| PVT-V2-B5        |  224 |  83.8 |     82.0    | [config](configs/pvt_v2/pvt_v2_b5.py)    | 313M [[Google]](https://drive.google.com/file/d/1TKQIdpOFoFs9H6aApUNJKDUK95l_gWy0/view?usp=sharing) [[GitHub]](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b5.pth)|

- PVTv1 on ImageNet-1K

| Method     | Size | Acc@1 | #Params (M) | Config                                             | Download                                                                                   |
|------------|:----:|:-----:|:-----------:|----------------------------------------------------|--------------------------------------------------------------------------------------------|
| PVT-Tiny   |  224 |  75.1 |     13.2    | [config](classification/configs/pvt/pvt_tiny.py)   | 51M [[Google]](https://drive.google.com/file/d/1yau8uMRl-mnlTAUn4I7vypss3wjVltt5/view?usp=sharing) [[GitHub]](https://github.com/whai362/PVT/releases/download/v2/pvt_tiny.pth) |
| PVT-Small  |  224 |  79.8 |     24.5    | [config](classification/configs/pvt/pvt_small.py)  | 93M [[Google]](https://drive.google.com/file/d/1ds9Rb9wRh9IzGV0CZMM0hnS0QAM_qyIF/view?usp=sharing) [[GitHub]](https://github.com/whai362/PVT/releases/download/v2/pvt_small.pth) |
| PVT-Medium |  224 |  81.2 |     44.2    | [config](classification/configs/pvt/pvt_medium.py) | 168M [[Google]](https://drive.google.com/file/d/1c2EkzszygPET83h-w4eh-Ef4V_d1a8kw/view?usp=sharing) [[GitHub]](https://github.com/whai362/PVT/releases/download/v2/pvt_medium.pth)|
| PVT-Large  |  224 |  81.7 |     61.4    | [config](classification/configs/pvt/pvt_large.py)  | 234M [[Google]](https://drive.google.com/file/d/1C07_swTQeWvppIzQrl_0H7UDk4SsalkJ/view?usp=sharing) [[GitHub]](https://github.com/whai362/PVT/releases/download/v2/pvt_large.pth)|

## Evaluation
To evaluate a pre-trained PVT-Small on ImageNet val with a single GPU run:
```
sh dist_train.sh configs/pvt/pvt_small.py 1 --data-path /path/to/imagenet --resume /path/to/checkpoint_file --eval
```
This should give
```
* Acc@1 79.764 Acc@5 94.950 loss 0.885
Accuracy of the network on the 50000 test images: 79.8%
```

## Training
To train PVT-Small on ImageNet on a single node with 8 gpus for 300 epochs run:

```
sh dist_train.sh configs/pvt/pvt_small.py 8 --data-path /path/to/imagenet
```

## Calculating FLOPS & Params

```
python get_flops.py pvt_v2_b2
```
This should give
```
Input shape: (3, 224, 224)
Flops: 4.04 GFLOPs
Params: 25.36 M
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
