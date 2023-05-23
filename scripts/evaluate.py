import openpifpaf.benchmark
import sys

MODELS = ['shufflenetv2k16', 'out/test.epoch250'] # TODO add the pruned checkpoints here

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