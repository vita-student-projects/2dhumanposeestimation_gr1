# DLAV 2023 Project: 2D Human Pose Estimation

**Group 1** (Sabri el Almani - Louis Gevers)

---

## TODO

Implementations:
- Implement dataset.py
- Implement inference.py
- Script for evaluation metrics

Training:
- swin_t train
- swin_s train
- resnet reference?

Writing:
- Contribution overview
- Experimental setup
- Results
- Conclusion
- References
- Instructions for running the models

## Contribution overview

TODO:

```
Highlight the changes you have made to existing work. Justify the reasons behind these changes (What do you want to improve? and How?). It’s better to use visuals to guide the reader to understand your work.
```

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

To avoid doing this manually, simply run the following script:

```bash
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

- OpenPifPaf, Swin

---

## Instructions for running the models

TODO:

```
Instructions to run and test model + links to download the models weights/checkpoints
```

All of our code can be found under the scripts folder:
- [scripts/dataset.py](scripts/dataset.py) for the CrowdPose dataset class
- [scripts/train.py](scripts/train.py) for training the models
- [scripts/inference.py](scripts/inference.py) for predictions