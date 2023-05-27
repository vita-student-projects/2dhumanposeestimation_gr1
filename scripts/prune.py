import torch
from torch.nn.utils import prune
from openpifpaf.network.factory import CHECKPOINT_URLS
import os
import argparse

def create_pruned_checkpoint(checkpoint, amount):
    checkpoint_name = os.path.splitext(checkpoint)[0]
    checkpoint = load_openpifpaf_checkpoint(checkpoint)
    model = checkpoint['model']
    to_prune = [(module, 'weight') for module in model.modules() if type(module) == torch.nn.Linear or type(module) == torch.nn.Conv2d]
    # Remove lowest amount% of weights
    prune.global_unstructured(to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    # Apply the pruning (yes the function name is counterintuitive)
    for module, name in to_prune:
        if prune.is_pruned(module):
            prune.remove(module, name)
    # Reset the epoch
    checkpoint['epoch'] = 0
    # Store the checkpoint back (the weights are modified in place)
    file_name = f'{checkpoint_name}_pruned_{amount}.pth'
    torch.save(checkpoint, file_name)
    print(f"Saved pruned model to {file_name}")

def load_openpifpaf_checkpoint(checkpoint):
    checkpoint = CHECKPOINT_URLS.get(checkpoint, checkpoint)
    if checkpoint.startswith('http'):
        print(f"Downloading checkpoint '{checkpoint}'")
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint,
            check_hash=not checkpoint.startswith('https'))
    else:
        checkpoint = torch.load(checkpoint)
    return checkpoint

def cli(parser = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', default=None, help='OpenPifPaf checkpoint to prune (local file or one supported by OpenPifPaf')
    parser.add_argument('-a', '--amount', default=0.2, help='Fraction of weights to prune', type=float)
    args = parser.parse_args()
    return args.checkpoint, args.amount


if __name__ == '__main__':
    checkpoint, amount = cli()
    create_pruned_checkpoint(checkpoint, amount)