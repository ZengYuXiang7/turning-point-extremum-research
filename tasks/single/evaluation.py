import torch

from tasks.single.power_evaluation import evaluate_power
from tasks.single.weather_evaluation import evaluate_weather


def test_checkpoint(
    args,
    model,
    repository,
    datasets,
    loaders,
    device,
    checkpoint,
    contract_checkpoint,
    best_epoch,
    best_mse,
    best_acc30,
    history,
    pretraining,
    model_uses_future_weather,
):
    # Single 程序恢复自己的检查点并进入对应测试分支。
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"], strict=args.model != "QwenMLP")
    model.eval()
    if args.scenario == "ForecastWeather":
        result = evaluate_weather(
            args, model, datasets, device, checkpoint, best_epoch, best_mse
        )
    else:
        result = evaluate_power(
            args,
            model,
            repository,
            datasets,
            loaders,
            device,
            checkpoint,
            contract_checkpoint,
            best_epoch,
            best_mse,
            best_acc30,
            history,
            pretraining,
            model_uses_future_weather,
        )
    return result
