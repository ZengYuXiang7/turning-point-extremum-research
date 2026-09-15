from functools import partial
import math

import torch



def cosine_hard_restarts_learning_rate(
    current_step, *, warmup_steps, training_steps, cycles, minimum_ratio,
):
    # 前10%线性热身后执行余弦硬重启
    if current_step < warmup_steps:
        ratio = float(current_step) / float(max(1, warmup_steps))
    else:
        progress = float(current_step - warmup_steps) / float(
            max(1, training_steps - warmup_steps)
        )
        ratio = minimum_ratio
        if progress < 1.0:
            angle = math.pi * ((float(cycles) * progress) % 1.0)
            ratio += (1.0 - minimum_ratio) * 0.5 * (1.0 + math.cos(angle))
    return ratio


def constant_learning_rate(current_step):
    ratio = 1.0
    return ratio


def build_pretrain_scheduler(args, optimizer, batches_per_epoch):
    # 建立当前项目的预训练学习率调度
    scheduler_steps_per_batch = False
    if args.pretrain_lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5,
        )
    if args.pretrain_lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.pretrain_epochs, eta_min=1e-6,
        )
    if args.pretrain_lr_scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01,
            total_iters=args.pretrain_epochs,
        )
    if args.pretrain_lr_scheduler == "cosine_hard_restarts":
        training_steps = batches_per_epoch * args.pretrain_epochs
        warmup_steps = 0.1 * training_steps
        learning_rate = partial(
            cosine_hard_restarts_learning_rate,
            warmup_steps=warmup_steps,
            training_steps=training_steps,
            cycles=1,
            minimum_ratio=0.01,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
        scheduler_steps_per_batch = True
    if args.pretrain_lr_scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, constant_learning_rate,
        )
    return scheduler, scheduler_steps_per_batch
