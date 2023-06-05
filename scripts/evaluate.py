import openpifpaf.benchmark
import sys

MODELS = ['resnet50-crowdpose', 'shufflenetv2k16', 'checkpoints/shufflenet_pruned1.pt', 'checkpoints/shufflenet_pruned2.pt', 'checkpoints/resnet_pruned.pt', 'checkpoints/swin.pt'] 

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
