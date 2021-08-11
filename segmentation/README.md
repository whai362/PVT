# Applying PVT to Semantic Segmentation

Here, we take [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0) as an example, applying PVT to SemanticFPN.

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


## Usage

Install MMSegmentation.


## Data preparation

Preparing ADE20K according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md#prepare-datasets) in MMSegmentation.


## Results and models

| Method       | Backbone   | Pretrain    | Iters | mIoU(code) | mIoU(paper) | Config                                                | Download                                                                                          |
|--------------|------------|-------------|:-----:|:----------:|:-----------:|-------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| Semantic FPN | PVT-Tiny   | ImageNet-1K |  40K  |    36.6    |     35.7    | [config](configs/sem_fpn/PVT/fpn_pvt_t_ade20k_40k.py) | [log](https://drive.google.com/file/d/18NodMVuLWSHGjbUz6oMbtDnV2EddQEkC/view?usp=sharing) & [model](https://drive.google.com/file/d/13SaiOJ9hH7Wwg_AyeQ158LNV9vtjq6Lu/view?usp=sharing) |
| Semantic FPN | PVT-Small  | ImageNet-1K |  40K  |    41.9    |     39.8    | [config](configs/sem_fpn/PVT/fpn_pvt_s_ade20k_40k.py) | [log](https://drive.google.com/file/d/12FnAEQHWFa5K0wurEn1LcI6BZD7vexJV/view?usp=sharing) & [model](https://drive.google.com/file/d/13fy-FXAfYnHgHRaUiJWVBON670wFLIiD/view?usp=sharing) |
| Semantic FPN | PVT-Medium | ImageNet-1K |  40K  |    43.5    |     41.6    | [config](configs/sem_fpn/PVT/fpn_pvt_m_ade20k_40k.py) | [log](https://drive.google.com/file/d/1yNQLCax2Qx7xOQVL0v84KwhcNkWbp_s8/view?usp=sharing) & [model](https://drive.google.com/file/d/10ErJJZCcucnjjo8et2ivuHRzxbwc04y2/view?usp=sharing) |
| Semantic FPN | PVT-Large  | ImageNet-1K |  40K  |    43.5    |     42.1    | [config](configs/sem_fpn/PVT/fpn_pvt_l_ade20k_40k.py) | [log](https://drive.google.com/file/d/11-gMPyz19ExtfT3Tp8P40EYUKHd11ntA/view?usp=sharing) & [model](https://drive.google.com/file/d/1JkaXbTorIWLj9Oe5Dh6kzH_1vtrRFJRL/view?usp=sharing) |

## Evaluation
To evaluate PVT-Small + Semantic FPN on a single node with 8 gpus run:
```
dist_test.sh configs/sem_fpn/PVT/fpn_pvt_s_ade20k_40k.py /path/to/checkpoint_file 8 --out results.pkl --eval mIoU
```


## Training
To train PVT-Small + Semantic FPN on a single node with 8 gpus run:

```
dist_train.sh configs/sem_fpn/PVT/fpn_pvt_s_ade20k_40k.py 8
```

# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
