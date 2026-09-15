import torch
from torch import nn
from tqdm import tqdm

from utils.acc30_loss import Acc30BoundaryLoss


def run_epoch(model, loader, device, optimizer, objective, dbloss_weight, train, power_means, power_stds, show_progress, epoch,):
    # 联合面板 epoch 始终读取16台历史并同时预测16台功率。
    model.train(train)
    total_loss = 0.0
    total_mse = 0.0
    points = 0
    strict_passed = 0
    strict_valid = 0

    if train:
        context = torch.enable_grad()
        stage = "train"
    else:
        context = torch.no_grad()
        stage = "val"

    batches = tqdm(loader, total=len(loader), desc=f"epoch {epoch} {stage}", disable=show_progress == 0, leave=False,)
    with context:
        for past, target, turbine_id, _ in batches:
            past = past.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            turbine_id = turbine_id.to(device, non_blocking=True)
            batch_mean = power_means[turbine_id].unsqueeze(-1)
            batch_std = power_stds[turbine_id].unsqueeze(-1)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",):
                prediction = model(past, turbine_id)
                mse = torch.mean((prediction - target) ** 2)
                if objective is None:
                    loss = mse
                elif isinstance(objective, Acc30BoundaryLoss):
                    loss = objective(prediction, target, batch_mean, batch_std)
                else:
                    power_dbloss = objective(prediction, target)
                    loss = mse + dbloss_weight * power_dbloss

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            count = int(target.numel())
            total_loss += float(loss.detach().cpu()) * count
            total_mse += float(mse.detach().cpu()) * count
            points += count

            # 在各风机训练尺度上计算严格 Acc30。
            with torch.no_grad():
                truth_kw = target * batch_std + batch_mean
                prediction_kw = prediction * batch_std + batch_mean
                valid = truth_kw > 100.0
                passed = valid & (
                    torch.abs(prediction_kw - truth_kw) <= 0.30 * truth_kw
                )
                strict_valid += int(valid.sum().item())
                strict_passed += int(passed.sum().item())

            batches.set_postfix(loss=total_loss / points, mse=total_mse / points, refresh=False,)

    strict_acc30 = 100.0 * strict_passed / strict_valid
    return total_loss / points, total_mse / points, strict_acc30
