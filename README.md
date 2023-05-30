# DLAV 2023 Project: 2D Human Pose Estimation

**Group 1** (Sabri el Almani - Louis Gevers)

---

## Introduction

Human pose estimation aims to detect and localise body joints of a person - such as elbows, wrists, and other articulations - from images or videos, to determine the pose of the full body. 
When selecting a human pose estimation method for autonomous driving, there are several factors to take into consideration. 
First, the chosen method must have a high level of accuracy to ensure safety, which can be achieved through the implementation of state-of-the-art approaches. 
Additionally, it should be capable of performing these estimations from video and do so in an online fashion. 
Finally, the method should perform well in urban environments, where it may encounter challenges such as crowds and occlusions.
Given these considerations, we chose [OpenPifPaf](https://github.com/openpifpaf/openpifpaf) [1] as the base method, and training and evaluating it on the [CrowdPose dataset](https://github.com/Jeff-sjtu/CrowdPose).

## Contribution overview

OpenPifPaf aims to be a suitable pose estimation model for autonomous vehicles [1].
To do so, it's important to consider the computation limit of an on-board computer of a vehicle, and its memory limit.
We therefore attempt to contribute to OpenPifPaf's work to make it faster, and if possible reduce the memory requirements.

To address this, we explored two different methods:
1. Training a new backbone
2. Pruning existing backbones

Furthermore, focussing on urban environments, we train and evaluate on the CrowdPose dataset.

### New backbone: Swin

OpenPifPaf provides several pretrained backbones on their [official torchhub](https://github.com/openpifpaf/torchhub/releases).
Inspired by how the equivalent performance of the ResNet backbone with the much smaller and faster ShuffleNet backbone, we aim to train  new backbone.

As all trained backbones provided by OpenPifPaf are convolutional networks, we try to explore a transformer-based backbone.
More specifically, we opted for Swin [2], a state-of-the-art general purpose vision transformer.

We explored [HRFormer](https://github.com/HRNet/HRFormer) initially, but due to integration issues and inactivity from the author since 2021, we ended up with the already available to train Swin backbone instead.

### Pruning ShuffleNet and ResNet

Pruning involves removing a fraction of weights with lowest magnitude before retraining the compressed network.
It has been shown to effectively reduce the number of parameters without losing accuracy of image classification networks [3].
They show that this method can achieve compression rates between 9 and 13 with equal performance, and 3x to 4x layer-wise speedup on CPU and GPU, making it particularly relevant for real-time applications such as autonomous driving.

We therefore propose a pruning script for OpenPifPaf backbones, and a single iteration of pruned ResNet and ShuffleNet trained backbones for reference.
Multiple iterations would achieve higher compression rates, yet due to time and resource limitations we reserve them for future work.

## Experimental setup

TODO:

```
What are the experiments you conducted? What are the evaluation
metrics?
```

## Dataset


We use the [CrowdPose dataset](https://github.com/Jeff-sjtu/CrowdPose).
The repository contains the Google drive links for downloading the images and annotations.
Images should be downloaded under the [/data-crowdpose/images/](/data-crowdpose/images/) directory, and annotations under [/data-crowdpose/json/](/data-crowdpose/json/).

To avoid doing this manually, simply run the following script in the data directory:

```bash
$ cd data-crowdpose
$ sh data.sh
```

Note: make sure to have `gdown` installed (provided in the `requirements.txt`).

## Results

TODO:

```
Qualitative and Quantitative results of your experiments
```

## Conclusion

TODO: short

## References

[1] S. Kreiss, L. Bertoni, and A. Alahi, “OpenPifPaf: Composite Fields for Semantic Keypoint Detection and Spatio-Temporal Association,” *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 8, pp. 13 498–13 511, Aug. 2021.

[2] Swin transformer: Hierarchical vision transformer using shifted windows,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, Oct. 2021, pp. 10 012–10 022.

---

## Instructions for running the models

TODO:

```
Instructions to run and test model + links to download the models weights/checkpoints
```
First, make sure to have downloaded the CrowdPose dataset in the `data-crowdpose` folder (see dataset section).
To download the model weights, go into the `checkpoints` directory and run the corresponding script:

```bash
$ cd checkpoints
$ sh checkpoints.sh
```

With the dataset and model weights ready, you can use all the remaining scripts in this repository.

All of our code can be found under the scripts folder:
- [scripts/dataset.py](scripts/dataset.py) for the CrowdPose dataset class
- [scripts/train.py](scripts/train.py) for training the models
- [scripts/inference.py](scripts/inference.py) for predictions