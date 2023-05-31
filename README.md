# DLAV 2023 Project: 2D Human Pose Estimation

**Group 1** (Sabri El Amrani - Louis Gevers)

---

## Introduction

Human pose estimation aims to detect and localise body joints of a person - such as elbows, wrists, and other articulations - from images or videos, to determine the pose of the full body. 
When selecting a human pose estimation method for autonomous driving, there are several factors to take into consideration. 
First, the chosen method must have a high level of accuracy to ensure safety, which can be achieved through the implementation of state-of-the-art approaches. 
Additionally, it should be capable of performing these estimations from video and do so in an online fashion. 
Finally, the method should perform well in urban environments, where it may encounter challenges such as crowds and occlusions.
Given these considerations, we chose [OpenPifPaf](https://github.com/openpifpaf/openpifpaf) [1] as the base method, and trained and evaluated it on the [CrowdPose dataset](https://github.com/Jeff-sjtu/CrowdPose).

## Contribution overview

OpenPifPaf aims to be a suitable pose estimation model for autonomous vehicles [1].
To do so, it's important to consider the computation limit of an on-board computer of a vehicle, and its memory limit.
We therefore attempt to contribute to OpenPifPaf's work to make it faster, and if possible reduce the memory requirements.

To address this, we explored two different methods:
1. Training a new backbone
2. Pruning existing backbones

Furthermore, focusing on urban environments, we trained and evaluated on the CrowdPose dataset.

<figure align="center">
  <img src="images/openpifpaf.JPG" alt="openpifpaf" width="800">
  <figcaption align="center">Qualitative visualisation of 2D pose estimation by OpenPifPaf [1].</figcaption>
</figure>

### New backbone: Swin

OpenPifPaf provides several pretrained backbones on their [official torchhub](https://github.com/openpifpaf/torchhub/releases).
Inspired by the equivalent performance of the ResNet backbone with the much smaller and faster ShuffleNet backbone, we aimed to train with a new backbone.

As all trained backbones provided by OpenPifPaf are convolutional networks, we tried to explore a transformer-based backbone.
More specifically, we opted for Swin [2], a state-of-the-art general purpose vision transformer.

We initially explored [HRFormer](https://github.com/HRNet/HRFormer), but due to integration issues and inactivity from the author since 2021, we stuck to the ready-to-train Swin backbone instead.

### Pruning ShuffleNet and ResNet

Pruning involves removing a fraction of weights with lowest magnitude before retraining the compressed network.
It has been shown to effectively reduce the number of parameters without losing accuracy of image classification networks [3].
They show that this method can achieve compression rates between 9 and 13 with equal performance, and 3x to 4x layer-wise speedup on CPU and GPU, making it particularly relevant for real-time applications such as autonomous driving.

We therefore propose a pruning script for OpenPifPaf backbones, and a single iteration of pruned ResNet and ShuffleNet trained backbones for reference.
Multiple iterations would achieve higher compression rates, yet due to time and resource limitations we reserve them for future work.

## Experimental setup

Aiming to extend OpenPifPaf, we evaluate our models with the same metrics the previous models have been evaluated on.
We will concentrate on keeping a similar average precision (AP), while having a lower inference time, and (as an ideal by-product) a lower file size (MB) of the model.

### Training with a Swin backbone

We trained OpenPifPaf with the `swin_t` (Swin tiny) backbone on the CrowdPose dataset.
Similar to the training of previous backbones on CrowdPose [1], we chose a learning rate of 1e-4, warmed up for an epoch, a weight decay of 1e-5, a batch size of 32, and ran for 250 epochs.

For the implementation of the training procedure, please refer to [scripts/train.py](scripts/train.py).

### Training pruned backbones

We pruned both the `resnet50` and `shufflenetv2k16` networks, removing 20% of the weights of their linear and convolutional layers.
We then trained them for 100 epochs with a learning rate of 1e-5.

For more details about the training implementation, please refer to [scripts/train.py](scripts/train.py).

### Compare with existing work

Currently, OpenPifPaf officially only provides a [ResNet backbone](https://openpifpaf.github.io/plugins_crowdpose.html#prediction) that has been trained on CrowdPose.
As its performance was similar to ShuffleNet on the COCO dataset, we will use this as our baseline.
We will also evaluate a ShuffleNet, but only to compare inference time and model size.

Overal, we aimed to stay as consistent as possible with the original work, attempting to not modify the existing library.

## Dataset


We use the [CrowdPose dataset](https://github.com/Jeff-sjtu/CrowdPose), which is already compatible with OpenPifPaf.
The repository contains the Google drive links for downloading the images and annotations.
Images should be downloaded under the [/data-crowdpose/images/](/data-crowdpose/images/) directory, and annotations under [/data-crowdpose/json/](/data-crowdpose/json/).

To avoid doing this manually, simply run the following script in the data directory:

```bash
$ cd data-crowdpose
$ sh data.sh
```

Note: make sure to have `gdown` installed (provided in the `requirements.txt`).

## Results

After running our experiments, we run OpenPifPaf's benchmark using the [scripts/evaluate.py](scripts/evaluate.py) script:

| Checkpoint                                |       AP |   AP0.5 |   AP0.75 |   APM |   APL |   t_total |   t_NN |   t_dec |    size |
|:------------------------------------|---------:|--------:|---------:|------:|------:|----------:|-------:|--------:|--------:|
| [shufflenetv2k16]                   |  n/a |       n/a |        n/a |     n/a |     n/a |      59ms |   26ms |    31ms |  38.9MB |
| [resnet50-crowdpose]                | __23.3__ |    43.9 |     20.1 |  16.2 |  25.2 |      47ms |   39ms |     5ms |  96.4MB |
| [checkpoints/swin.pt]               | __34.3__ |    57.9 |     32.9 |     6 |  42.5 |      38ms |   29ms |     6ms | 105.8MB |
| [checkpoints/resnet_pruned.pt]      | __40.5__ |    64.7 |     38.9 |  23.2 |  45.7 |      45ms |   33ms |     9ms |  91.5MB |
| [checkpoints/shufflenet_pruned2.pt] | __48.1__ |    72.1 |       48 |  21.5 |  55.2 |      36ms |   25ms |     9ms |  34.9MB |

Remember that the `shufflenetv2k16` backbone was not trained on CrowPose, and was not able to give use any good results due incompatible heads, we therefore do not report its AP.
We do see however that the inference of its pruned version is slightly faster, as well as being more lightweight.

A second observation that can be made is that the Swin backbone seems to reduce inference time. It makes for a heavier network and its AP is not really good, however.

Finally, the pruned ResNet is slighlty faster and more leightweight than its counterpart ResNet50-Crowdpose, a promising result given the aim we set ourselves. The AP of ResNet50-CrowdPose and our pruned ResNet should probably not be compared, however: we pruned models pretrained on COCO, which gives them an unfair advantage.

## Conclusion

In conclusion, the results we obtained from our two contributions show some promise, and they probably deserve more exploration with the hope of reaching state-of-the-art performance.

First, the Swin backbone improves inference time, in spite of taking more space in memory. Its precision is not really satisfactory, but we used the same hyperparameters as for ShuffleNet: that was probably not the most appropriate choice given the radically different architectures of the two backbones.

The pruning definitely shows promise as well: it has the double benefit of speeding up inference and reducing model size. More pruning iterations and a sparser representation (if the hardware supports it) would probably further enhance those qualities
To reach higher AP we should probably start by exploring the learning hyperparameter space to find better suited candidates. It would probably be worth training the model on COCO as well to get higher performance.

Finally, we should look for better baselines to assess our contibutions: the absence of results for SuffleNetv2k16 and the suprisingly low AP for ResNet50-CrowdPose make it difficult to evaluate the impact of backbone change and pruning on performance.   

## References

[1] S. Kreiss, L. Bertoni, and A. Alahi, “OpenPifPaf: Composite Fields for Semantic Keypoint Detection and Spatio-Temporal Association,” *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 8, pp. 13 498–13 511, Aug. 2021.

[2] Swin transformer: Hierarchical vision transformer using shifted windows,” in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, Oct. 2021, pp. 10 012–10 022.

[3] S. Han, H. Mao, and W. J. Dally, Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding, Feb. 2016.

---

## Instructions for running the models

First, make sure to have downloaded the CrowdPose dataset in the `data-crowdpose` folder (see dataset section).
To download the model weights, go into the `checkpoints` directory and run the corresponding script:

```bash
$ cd checkpoints
$ sh checkpoints.sh
```

With the dataset and model weights ready, you can use all the remaining scripts in this repository.

### Training

Preset training script: [scripts/train.py](scripts/train.py)

```bash
$ python scripts/train.py --checkpoint <your-checkpoint> --output <outputs-prefix>
```

Additionally, all [openpifpaf.train](https://openpifpaf.github.io/cli_help.html#train) options are also available.

### Pruning

Prune a checkpoint and save it in another file: [scripts/prune.py](scripts/prune.py)

```bash
python scripts/prune.py --checkpoint <checkpoint-to-prune> --amount <percentage-to-prune>
```

### Inference

Infer using the best performing pruned shufflenet checkpoint: [scripts/inference.py](scripts/inference.py)

```bash
python scripts/inference.py <image> --image-output <output-image> --json-output <output-annotation>
```

Additionally, all [openpifpaf.predict](https://openpifpaf.github.io/cli_help.html#predict) options are also available.

### Evaluating

Evaluate all checkpoints of interest of our project, to generate the table for the results: [scripts/evaluate.py](scripts/evaluate.py)

```bash
$ python scripts/evaluate.py
```

Additionally, all [openpifpaf.benchmark](https://openpifpaf.github.io/cli_help.html#benchmark) options are also available.

### SCITAS

Some useful run files for SCITAS are made available as well under the `sbatch` folder:
- [evaluate.run](sbatch/evaluate.run) = run the evaluation script
- [train-pruned-resnet.run](sbatch/train-pruned-resnet.run) = prune and train a ResNet backbone
- [train-pruned-shufflenet.run](sbatch/train-pruned-shufflenet.run) = prune and train a ShuffleNet backbone
- [train-swin.run](sbatch/train-swin.run) = train a Swin backbone (make sure to update the main of the `train.py` file)
