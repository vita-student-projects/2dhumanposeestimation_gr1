import openpifpaf.predict
import sys

# Run the benchmarks on all models of interest
def run():
    # Use the pruned checkpoint
    args = f"--checkpoint checkpoints/shufflenet_pruned2.pt"
    # Add arguments to system
    sys.argv.extend(args.split())
    # Run benchmark
    openpifpaf.predict.main()

if __name__ == "__main__":
    run()