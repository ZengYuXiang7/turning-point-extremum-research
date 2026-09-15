import torch
from torch import nn
from tqdm import tqdm


class MaskedMarketReconstruction(nn.Module):
    """以 SATRA 的完整 SPE+DRT 骨干重建随机遮蔽的风机时间 token。"""

    def __init__(self, forecast_model: nn.Module) -> None:
        super().__init__()
        self.backbone = forecast_model
        self.mask_token = nn.Parameter(torch.empty(1, 1, 1, forecast_model.hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder = nn.Sequential(
            nn.Linear(forecast_model.hidden_dim, forecast_model.hidden_dim),
            nn.GELU(),
            nn.Linear(forecast_model.hidden_dim, forecast_model.channels),
        )

    def forward(self, past: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        representation = self.backbone.encode_masked_history(
            past,
            token_mask,
            self.mask_token,
        )
        reconstruction = self.decoder(representation)
        return reconstruction

def sample_panel_token_mask(
    past: torch.Tensor,
    mask_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """独立随机遮蔽每个风机、每个历史时刻的完整特征 token。"""
    random_values = torch.rand(
        past.shape[:3],
        generator=generator,
        device=past.device,
    )
    token_mask = random_values < mask_ratio
    return token_mask

def run_satra_masked_epoch(
    model,
    loader,
    optimizer,
    device,
    mask_ratio,
    mask_seed,
    train,
    show_progress,
    epoch,
    scheduler,
    scheduler_steps_per_batch,
):
    # 验证阶段每轮重置随机源，保证各轮使用同一组遮蔽。
    model.train(train)
    generator = torch.Generator(device=device)
    generator.manual_seed(mask_seed)
    if train:
        stage = "train"
    else:
        stage = "val"
    batches = tqdm(
        loader,
        total=len(loader),
        desc=f"SATRA pretrain {epoch} {stage}",
        disable=show_progress == 0,
        leave=False,
    )
    error_sum = 0.0
    masked_features = 0
    with torch.set_grad_enabled(train):
        for past, _, _, _ in batches:
            past = past.to(device, non_blocking=True)
            token_mask = sample_panel_token_mask(past, mask_ratio, generator)
            feature_mask = token_mask.unsqueeze(-1).expand_as(past)
            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                reconstruction = model(past, token_mask)
            loss = (reconstruction.float() - past).square().masked_select(feature_mask).mean()
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler_steps_per_batch:
                    scheduler.step()

            count = int(feature_mask.sum().item())
            error_sum += float(loss.detach().cpu()) * count
            masked_features += count
            batches.set_postfix(MSE=error_sum / masked_features, refresh=False)

    mse = error_sum / masked_features
    return mse
