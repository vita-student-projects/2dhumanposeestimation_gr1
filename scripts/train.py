import openpifpaf
import openpifpaf.train
import sys

# Wrapper function to make the call to OpenPifPaf's training method easier
def train_openpifpaf(output="test", epochs: int = 1000, basenet: str = "resnet50"):
    args = [
        "--dataset",
        "crowdpose",
        "--output",
        f"out/{output}",
        "--epochs",
        str(epochs),
        "--basenet",
        basenet,
    ]
    sys.argv.extend(args)
    openpifpaf.train.main()


if __name__ == "__main__":
    train_openpifpaf()
