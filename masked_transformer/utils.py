
import torch
import torch.optim as optim

# =============================================================================
# def mask_latents(zq, indices, mask_prob=0.15):
#     B, L, D = zq.shape
#     mask = torch.rand(B, L, device=zq.device) < mask_prob
#     zq_masked = zq.clone()
#     zq_masked[mask] = 0.0
#     return zq_masked, indices, mask
# =============================================================================


import torch
import random

import torch
import random

def mask_latents(zq, indices, mask_prob=None):
    # If mask_prob isn't provided, choose random [0.5, 1.0]
    if mask_prob is None:
        mask_prob = random.uniform(0.5, 1.0)

    B, L, D = zq.shape
    mask = torch.rand(B, L, device=zq.device) < mask_prob

    zq_masked = zq.clone()
    zq_masked[mask] = 0.0

    return zq_masked, indices, mask, mask_prob



import math
import torch.optim as optim

def build_scheduler(optimizer, sched_type, total_steps, warmup_steps=0):
    if sched_type == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    elif sched_type == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    elif sched_type == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / max(1, total_steps)))

    elif sched_type == "linear_warmup":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, 1 - (step - warmup_steps) / max(1, (total_steps - warmup_steps)))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif sched_type == "cosine":
        # cosine decay from 1.0 -> 0.0 after warmup
        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif sched_type == "cosine_warmup":
        # alias if you prefer explicit naming
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}")


# =============================================================================
# def build_scheduler(optimizer, sched_type, total_steps, warmup_steps=0):
#     if sched_type == "constant":
#         return optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
#     elif sched_type == "step":
#         return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
#     elif sched_type == "linear":
#         return optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / total_steps)
#     elif sched_type == "linear_warmup":
#         def lr_lambda(step):
#             if step < warmup_steps:
#                 return step / max(1, warmup_steps)
#             return max(0.0, 1 - (step - warmup_steps) / (total_steps - warmup_steps))
#         return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
#     else:
#         raise ValueError(f"Unknown scheduler type: {sched_type}")
# =============================================================================
