import openpifpaf.benchmark
import sys

MODELS = ['checkpoints/shufflenet_pruned1.pt', 'checkpoints/shufflenet_pruned2.pt', 'checkpoints/resnet_pruned.pt', 'checkpoints/swin.pt', 'shufflenetv2k16', 'resnet50']

# Run the benchmarks on all models of interest
def run():
    checkpoints = " ".join(MODELS)
    # Arguments for openpifpaf
    args = f"--dataset crowdpose --checkpoints {checkpoints}"
    # Add arguments to system
    sys.argv.extend(args.split())
    # Run benchmark
    openpifpaf.benchmark.main()

if __name__ == "__main__":
    run()
