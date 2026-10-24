import torch
import os

NUM_ROUNDS = 10
NUM_WORKERS = 12

_SNAPSHOT_CACHE = {}
_TRACED_CACHE = {}


def init_worker():
    torch.set_num_threads(1)


def get_cached_model(path, model_cls, device: str | torch.device = "cpu"):
    key = (path, str(device))
    if key not in _SNAPSHOT_CACHE:
        model = model_cls().to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _SNAPSHOT_CACHE[key] = model
    return _SNAPSHOT_CACHE[key]


def get_cached_traced_model(
    path, model_cls, example_batch_size=8, device: str | torch.device = "cpu"
):
    key = (path, str(device))
    if key not in _TRACED_CACHE:
        model = get_cached_model(path, model_cls, device)
        first_linear = next(
            m for m in model.modules() if isinstance(m, torch.nn.Linear)
        )
        input_dim = first_linear.in_features
        example = torch.zeros(
            example_batch_size, input_dim, dtype=torch.float32, device=device
        )
        with torch.no_grad():
            traced = torch.jit.trace(model, example)
            traced = torch.jit.freeze(traced)
        _TRACED_CACHE[key] = traced
        _SNAPSHOT_CACHE.pop(key, None)
    return _TRACED_CACHE[key]


def parse_batch_num(filepath):
    try:
        filename = os.path.basename(filepath)
        parts = filename.replace(".pt", "").split("_")
        return int(parts[-1])
    except (IndexError, ValueError):
        return 0
