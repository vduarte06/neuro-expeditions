"""Pure-PyTorch stand-in for the compiled `MultiScaleDeformableAttention` CUDA extension used by DINO / Deformable
DETR. Same maths as the reference `ms_deform_attn_core_pytorch` in Deformable DETR, so it runs on CPU or Apple MPS.
Inference only (no backward). Registered in sys.modules by models/dinoeeg.py before the DINO-EEG code is imported.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def ms_deform_attn_forward(value, value_spatial_shapes, value_level_start_index, sampling_locations,
                           attention_weights, im2col_step):  # fmt: skip
    n, _, m, d = value.shape
    _, lq, m, levels, points, _ = sampling_locations.shape
    shapes = [(int(h), int(w)) for h, w in value_spatial_shapes]
    value_list = value.split([h * w for h, w in shapes], dim=1)
    grids = 2 * sampling_locations - 1
    sampled = []
    for lid, (h, w) in enumerate(shapes):
        v = value_list[lid].flatten(2).transpose(1, 2).reshape(n * m, d, h, w)
        g = grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    attn = attention_weights.transpose(1, 2).reshape(n * m, 1, lq, levels * points)
    out = (torch.stack(sampled, dim=-2).flatten(-2) * attn).sum(-1).view(n, m * d, lq)
    return out.transpose(1, 2).contiguous()


def ms_deform_attn_backward(*args, **kwargs):
    raise NotImplementedError("Inference-only stand-in; training needs the compiled CUDA extension.")
