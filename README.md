# Self-supervised Depth and Ego-motion Estimation for Monocular Thermal Video using Multi-spectral Consistency Loss

This github implements the system described in the paper:

 >Self-supervised Depth and Ego-motion Estimation for Monocular Thermal Video using Multi-spectral Consistency Loss
 >
 >[Ukcheol Shin](https://ukcheolshin.github.io/), Kyunghyun Lee, Seokju Lee, In So Kweon
 >
 >**Robotics and Automation Letter 2022 & ICRA 2022**
 >
 >[[PDF](https://arxiv.org/abs/2103.00760)] [[Project webpage](https://sites.google.com/view/t-sfmlearner)] [[Full paper](https://arxiv.org/abs/2103.00760)] [[Youtube](https://youtu.be/qIBcOuLYr70)] 

## Depth estimation results on ViViD dataset
[![Video Label](https://img.youtube.com/vi/qIBcOuLYr70/0.jpg)](https://youtu.be/qIBcOuLYr70)

## Prerequisite
This codebase was developed and tested with python 3.7, Pytorch 1.5.1, and CUDA 10.2 on Ubuntu 16.04. 

```bash
conda env create --file environment.yml
```

## Datasets

See "scripts/run_prepare_vivid_data.sh".

For ViViD Raw dataset, download the dataset provided on the [official website](https://sites.google.com/view/dgbicra2019-vivid/).

For our post-processed dataset and pre-trained models, you can download after fill out a simple [survey](https://docs.google.com/forms/d/e/1FAIpQLSd2IndM_BvsBQ2NypmoF8hGNdVFLQcdHifbHFYAgl62K_z-Pw/viewform?usp=pp_url).

We will send you an e-mail with a download link.

After download post-processed dataset, generate training/testing dataset

```bash
sh scripts/run_prepare_vivid_data.sh
```

## Training

The "scripts" folder provides several examples for training and testing.

You can train the depth and pose model on vivid dataset by running
```bash
sh scripts/train_vivid_resnet18_indoor.sh
sh scripts/train_vivid_resnet18_outdoor.sh
```
Then you can start a `tensorboard` session in this folder by
```bash
tensorboard --logdir=checkpoints/
```
and visualize the training progress by opening [https://localhost:6006](https://localhost:6006) on your browser. 


## Evaluation

You can evaluate depth and pose by running
```bash
sh scripts/test_vivid_indoor.sh
sh scripts/test_vivid_outdoor.sh
```
and visualize depth by running
```bash
sh scripts/run_inference.sh
```


### Depth Results 

#### Indoor 

|   Models   | Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|------------|---------|--------|-------|-----------|-------|-------|-------|
| Ours(T)    | 0.231   | 0.215  | 0.730 | 0.266     | 0.616 | 0.912 | 0.990 |
| Ours(MS)   | 0.163   | 0.123  | 0.553 | 0.204     | 0.771 | 0.970 | 0.995 |

#### Outdoor 

|   Models   | Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|------------|---------|--------|-------|-----------|-------|-------|-------|
| Ours(T)    | 0.157   | 1.179  | 5.802 | 0.211     | 0.750 | 0.948 | 0.985 |
| Ours(MS)   | 0.146   | 0.873  | 4.697 | 0.184     | 0.801 | 0.973 | 0.993 |


### Pose Estimation Results 

#### Indoor-static-dark

|Metric               | ATE     | RE      |
|---------------------|---------|---------|
| Ours(T)             | 0.0063  | 0.0092  |
| Ours(MS)            | 0.0057  | 0.0089  | 

#### Outdoor-night1

|Metric               | ATE     | RE      |
|---------------------|---------|---------|
| Ours(T)             | 0.0571  | 0.0280  |
| Ours(MS)            | 0.0562  | 0.0287  | 


## Citation
Please cite the following paper if you use our work, parts of this code, and pre-processed dataset in your research.
 
    @article{shin2021unsupervised,
      title={Unsupervised Depth and Ego-motion Estimation for Monocular Thermal Video using Multi-spectral Consistency Loss},
      author={Shin, Ukcheol and Lee, Kyunghyun and Lee, Seokju and Kweon, In So},
      journal={arXiv preprint arXiv:2103.00760},
      year={2021} 
    }



 ## Related projects
 
 * [SfMLearner-Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch) (CVPR 2017)
 * [SC-SfMLearner-Pytorch](https://github.com/JiawangBian/SC-SfMLearner-Release) (NeurIPS 2019)
