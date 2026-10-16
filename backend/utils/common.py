import torch
import os

NUM_ROUNDS = 10
NUM_WORKERS = 12

_SNAPSHOT_CACHE = {}
_TRACED_CACHE = {}


def init_worker():
    torch.set_num_threads(1)


def get_cached_model(path, model_cls):
    if path not in _SNAPSHOT_CACHE:
        model = model_cls().to("cpu")
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        _SNAPSHOT_CACHE[path] = model
    return _SNAPSHOT_CACHE[path]


def get_cached_traced_model(path, model_cls, example_batch_size=8):
    if path not in _TRACED_CACHE:
        model = get_cached_model(path, model_cls)
        first_linear = next(
            m for m in model.modules() if isinstance(m, torch.nn.Linear)
        )
        input_dim = first_linear.in_features
        example = torch.zeros(example_batch_size, input_dim, dtype=torch.float32)
        with torch.no_grad():
            traced = torch.jit.trace(model, example)
            traced = torch.jit.freeze(traced)
        _TRACED_CACHE[path] = traced
        _SNAPSHOT_CACHE.pop(path, None)
    return _TRACED_CACHE[path]


def parse_batch_num(filepath):
    try:
        filename = os.path.basename(filepath)
        parts = filename.replace(".pt", "").split("_")
        return int(parts[-1])
    except (IndexError, ValueError):
        return 0
