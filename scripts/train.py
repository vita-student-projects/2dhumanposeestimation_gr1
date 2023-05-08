from typing import Literal
import openpifpaf
import openpifpaf.train
import sys


def main(
    output: str,
    backbone: Literal["swin"],
    epochs: int = 300,  # default for crowdpose
):
    # NOTE: all parameters from OpenPifPaf paper for training on crowdpose

    # Base arguments
    args = f"--dataset crowdpose --output out/{output} --epochs {epochs}"
    # Optimizer
    args += " --momentum 0.95 --weight-decay 1e-5 --batch-size 32"
    # Learning rate scheduler
    args += " --lr 1e-3 --lr-decay 250 280 --lr-decay-epochs 10 --lr-decay-factor 10 --lr-warm-up-start-epoch 0 --lr-warm-up-epochs 1 --lr-warm-up-factor 1e-3"

    # Backbone specific arguments
    if backbone == "swin_s":
        args += " --basenet swin_b"
    elif backbone == "swin_b":
        args += " --basenet swin_b"
    elif backbone == "resnet":
        args += " --basenet resnet50"
    else:
        raise ValueError(f"Unknown backbone option: {backbone}")

    # Add arguments to system for openpifpaf script
    sys.argv.extend(args.split())
    # Run openpifpaf training script
    openpifpaf.train.main()


if __name__ == "__main__":
    main("test", backbone="swin")
