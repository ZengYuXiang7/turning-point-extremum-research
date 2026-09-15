from torch import nn

from models.multi_turbine_impl.multi_turbine import MultiTurbineBackbone
from models.multi_turbine_impl.stockecho import StockEchoNoFutureWeather


def select_backbone(args, repository, relation_weight):
    # 多风机任务显式选择联合面板骨干。
    if args.model == "MultiTurbine":
        backbone = MultiTurbineBackbone(
            channels=len(repository.feature_names),
            history_steps=args.history_steps,
            horizon=args.horizon_steps,
            relation_weight=relation_weight,
            use_dtw_prior=bool(args.satra_use_dtw_prior),
            hidden_dim=args.satra_hidden_dim,
            depth=args.satra_depth,
            kernel_size=args.satra_kernel_size,
            heads=args.satra_heads,
            tower_layers=args.satra_tower_layers,
            dropout=args.satra_dropout,
            expert_top_k=args.satra_expert_top_k,
        )
    elif args.model == "StockEcho":
        backbone = StockEchoNoFutureWeather(args.horizon_steps)
    return backbone


class MultiTurbineModel(nn.Module):
    """多风机任务的顶层模型。"""

    def __init__(self, args, repository, relation_weight) -> None:
        super().__init__()
        self.backbone = select_backbone(args, repository, relation_weight)

    def forward(self, *inputs):
        return self.backbone(*inputs)
