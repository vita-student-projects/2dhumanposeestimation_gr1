import torch
from hrformer.hrt import HRT


def create_base(weights: None) -> HRT:
    backbone = HRT(
        extra=dict(
            drop_path_rate=0.3,
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block="BOTTLENECK",
                num_blocks=(2,),
                num_channels=(64,),
                num_heads=[2],
                num_mlp_ratios=[4],
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block="TRANSFORMER_BLOCK",
                num_blocks=(2, 2),
                num_channels=(78, 156),
                num_heads=[2, 4],
                num_mlp_ratios=[4, 4],
                num_window_sizes=[7, 7],
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="TRANSFORMER_BLOCK",
                num_blocks=(2, 2, 2),
                num_channels=(78, 156, 312),
                num_heads=[2, 4, 8],
                num_mlp_ratios=[4, 4, 4],
                num_window_sizes=[7, 7, 7],
            ),
            stage4=dict(
                num_modules=2,
                num_branches=4,
                block="TRANSFORMER_BLOCK",
                num_blocks=(2, 2, 2, 2),
                num_channels=(78, 156, 312, 624),
                num_heads=[2, 4, 8, 16],
                num_mlp_ratios=[4, 4, 4, 4],
                num_window_sizes=[7, 7, 7, 7],
            ),
        ),
    )
    if weights is not None:
        print("Loading weights file")
        state = torch.load(weights)
        print("Loading weights into backbone")
        backbone.load_state_dict(state)
    return backbone


base = create_base(weights="./weights/hrt_base_imagenet_pretrained_top1_828.pth")

print("Done!")
print(base)
