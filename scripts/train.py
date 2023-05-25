import openpifpaf
import openpifpaf.train
import sys
import argparse

# Train utility for providing fixed arguments to OpenPifPaf
# This considerably reduces the training arguments required (and makes sure they're the same across runs).
# NOTE: The parameters have been chosen to train a pruned network
def train_pruned():
    # Base arguments
    args = f"--dataset crowdpose --epochs 100 --loader-workers 1 --fix-batch-norm"
    # Optimizer
    args += " --lr 1e-5 --momentum 0.95 --weight-decay 1e-5 --batch-size 32 " 

    # Add arguments to system for openpifpaf script
    sys.argv.extend(args.split())
    # Run openpifpaf training script
    openpifpaf.train.main()

# Normal training on CrowdPose. This is old code that was used to train swin...
def normal(
    output: str,
    backbone,
    epochs: int = 300,  # default for crowdpose
):
    # NOTE: all parameters from OpenPifPaf paper for training on crowdpose

    # Base arguments
    args = f"--dataset crowdpose --output out/{output} --epochs {epochs} --loader-workers 1"
    # Optimizer
    args += " --momentum 0.95 --weight-decay 1e-5 --batch-size 32" 
    # Learning rate scheduler
    args += " --lr 1e-4 --lr-decay 250 280 --lr-decay-epochs 10 --lr-decay-factor 10 --lr-warm-up-start-epoch 0 --lr-warm-up-epochs 1 --lr-warm-up-factor 1e-3"

    # Backbone specific arguments
    if backbone == "swin_s":
        args += " --basenet swin_b"
    elif backbone == "swin_b":
        args += " --basenet swin_b"
    elif backbone == "swin_t":
        args += " --basenet swin_t"
    elif backbone == "resnet":
        args += " --basenet resnet50"
    else: # We're using a checkpoint instead
        args += f" --checkpoint {backbone}"

    # Add arguments to system for openpifpaf script
    sys.argv.extend(args.split())
    # Run openpifpaf training script
    openpifpaf.train.main()


if __name__ == "__main__":
    train_pruned()
