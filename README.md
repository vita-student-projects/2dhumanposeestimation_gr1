# DLAV 2023 Project: 2D Human Pose Estimation

**Group 1** (Sabri el Almani - Louis Gevers)

---

TODO: short description of the project at the end

# 1. Environment

This project makes use of Docker to ensure a consistent environment setting.
In order to properly run the files, please make use of the provided `louisgevers/dlav2023` image which is available on [Dockerhub](https://hub.docker.com/r/louisgevers/dlav2023).

To run the container with an interactive command line in order to run the different scripts, run the following:

```bash
$ docker run -it --rm -v $(pwd):/usr/src louisgevers/dlav2023 sh
```

## Building the image yourself

If you prefer to build the image yourself, you can do so with the Dockerfile provided with project before running the command above:
```bash
$ docker build -t louisgevers/dlav2023 .
```

# 2. Dataset

We use the [CrowdPose dataset](https://github.com/Jeff-sjtu/CrowdPose).
The repository contains the Google drive links for downloading the images and annotations.
Images should be downloaded under the [/data-crowdpose/images/](/data-crowdpose/images/) directory, and annotations under [/data-crowdpose/json/](/data-crowdpose/json/).

To avoid doing this manually, simply run the following script:

```bash
$ sh data.sh
```

Note: run this in your container or install `gdown` with pip.

# TODO

instructions to run and test model + links to download the models weights/checkpoints

**Backbones to test**

- Pruning with big backbone (HRFormer) vs using a small backbone (second is extra) (we can compare performance with mobilenet)