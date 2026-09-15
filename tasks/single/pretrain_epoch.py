import math

import torch
from torch import nn
from tqdm import tqdm

def sample_token_mask(past, mask_ratio, generator):
    # 每个样本随机选择时间点，同一时间点的全部变量一起遮蔽
    batch_size, history_steps, channels = past.shape
    masked_count = math.ceil(history_steps * mask_ratio)
    noise = torch.rand(
        batch_size, history_steps,
        generator=generator, device=past.device,
    )
    ranks = noise.argsort(dim=-1).argsort(dim=-1)
    token_mask = ranks < masked_count
    mask = token_mask.unsqueeze(-1).expand(-1, -1, channels)
    return mask


class MaskedTokenPretraining(nn.Module):
    """共享预测编码器，重建头只在热身阶段使用。"""

    def __init__(self, forecast_model, history_steps):
        super().__init__()
        self.backbone = forecast_model
        self.reconstruction_head = nn.Linear(
            forecast_model.output_projection.in_features, history_steps,
        )

    def forward(self, past, turbine_id, mask):
        # 仅用可见位置估计每个变量的窗口统计量
        visible_count = (~mask).sum(dim=1, keepdim=True)
        visible_values = past.masked_fill(mask, 0.0)
        instance_mean = (visible_values.sum(dim=1, keepdim=True) / visible_count).detach()
        centered = (visible_values - instance_mean).masked_fill(mask, 0.0)
        instance_std = torch.sqrt(
            centered.square().sum(dim=1, keepdim=True) / visible_count
            + self.backbone.revin.eps
        ).detach()

        # 在重叠 patch 编码前清除被遮蔽位置的信息
        normalized = centered / instance_std
        normalized = normalized * self.backbone.revin.affine_weight
        normalized = normalized + self.backbone.revin.affine_bias
        normalized = normalized.masked_fill(mask, 0.0)
        encoded = self.backbone.encode_patches(normalized, turbine_id)
        reconstruction = self.reconstruction_head(encoded).transpose(1, 2)

        # 重建误差与训练段 StandardScaler 的尺度一致
        reconstruction = self.backbone.revin.denormalize(
            reconstruction, instance_mean, instance_std,
        )
        
        return reconstruction


def run_masked_epoch(
    model, loader, optimizer, device, mask_ratio,
    mask_seed, train, show_progress, epoch,
    scheduler, scheduler_steps_per_batch,
):
    # 验证每轮重置遮蔽随机源，确保比较使用同一组掩码
    model.train(train)
    generator = torch.Generator(device=device)
    generator.manual_seed(mask_seed)
    stage = "val"
    if train:
        stage = "train"
    batches = tqdm(
        loader, total=len(loader), desc=f"pretrain {epoch} {stage}",
        disable=show_progress == 0, leave=False,
    )

    # 标签和预测区间不参与自监督优化
    error_sum = 0.0
    masked_points = 0
    with torch.set_grad_enabled(train):
        for past, _, turbine_id, _ in batches:
            past = past.to(device, non_blocking=True)
            turbine_id = turbine_id.to(device, non_blocking=True)
            mask = sample_token_mask(past, mask_ratio, generator)
            if train:
                optimizer.zero_grad(set_to_none=True)

            # 仅对被遮蔽位置计算重建 MSE
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                reconstruction = model(past, turbine_id, mask)
            squared_error = (reconstruction.float() - past).square()
            loss = squared_error.masked_select(mask).mean()
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler_steps_per_batch:
                    scheduler.step()

            # 按真实遮蔽元素数聚合
            count = int(mask.sum().item())
            error_sum += loss.detach().item() * count
            masked_points += count
            batches.set_postfix(MSE=error_sum / masked_points, refresh=False)

    mse = error_sum / masked_points
    return mse
