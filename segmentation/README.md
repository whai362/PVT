# Applying PVT to Semantic Segmentation

Here, we take [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0) as an example, applying PVT to SemanticFPN.

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


## Usage

Install MMSegmentation.


## Data preparation

First, prepare ADE20K according to the guidelines in MMSegmentation.

Then, download the [weights](https://github.com/whai362/PVT/blob/main/README.md) pretrained on ImageNet, and put them in a folder `pretrained/`

## Results and models

|    Backbone     | Iters | mIoU(code) | mIoU(paper) | Config | Download  |
| :-------------: | :-----: | :-----: | :------: | :------------: | :----: |
|    PVT-Tiny     | 40K | 36.6    | 35.7 |  [config](https://github.com/whai362/PVT/blob/main/segmentation/configs/sem_fpn/PVT/fpn_pvt_t_ade20k_40k.py)  | [model&log](https://drive.google.com/drive/folders/18O7n7vi9WzD9DkuHFvFZSF31QVaKZ4AS?usp=sharing) |
|    PVT-Small    | 40K | 41.9    | 39.8 |  [config](https://github.com/whai362/PVT/blob/main/segmentation/configs/sem_fpn/PVT/fpn_pvt_s_ade20k_40k.py)  | [model&log](https://drive.google.com/drive/folders/18O7n7vi9WzD9DkuHFvFZSF31QVaKZ4AS?usp=sharing) |
|    PVT-Medium   | 40K | 43.5    | 41.6 |  [config](https://github.com/whai362/PVT/blob/main/segmentation/configs/sem_fpn/PVT/fpn_pvt_m_ade20k_40k.py)  | [model&log](https://drive.google.com/drive/folders/18O7n7vi9WzD9DkuHFvFZSF31QVaKZ4AS?usp=sharing) |
|    PVT-Large    | 40K | 43.5    | 42.1 |  [config](https://github.com/whai362/PVT/blob/main/segmentation/configs/sem_fpn/PVT/fpn_pvt_l_ade20k_40k.py)  | [model&log](https://drive.google.com/drive/folders/18O7n7vi9WzD9DkuHFvFZSF31QVaKZ4AS?usp=sharing) |

## Evaluation
To evaluate PVT-Small + SemFPN on a single node with 8 gpus run:
```
dist_test.sh configs/sem_fpn/PVT/fpn_pvt_s_ade20k_40k.py /path/to/checkpoint_file 8 --out results.pkl --eval mIoU
```


## Training
To train PVT-Small + SemFPN on a single node with 8 gpus run:

```
dist_train.sh configs/sem_fpn/PVT/fpn_pvt_s_ade20k_40k.py 8
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
