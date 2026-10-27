import torch
import os
import time
import queue
import contextlib
import numpy as np

GRADIENT_CLIP = 1.0


def _concat_pinned(arrays, dtype, pin):
    rows = sum(a.shape[0] for a in arrays)
    tail = arrays[0].shape[1:]
    out = torch.empty((rows, *tail), dtype=dtype, pin_memory=pin)
    np.concatenate(arrays, axis=0, out=out.numpy())
    return out


@contextlib.contextmanager
def _tf32(enabled):
    if not enabled:
        yield
        return
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


def train_on_batch(
    all_x,
    all_a,
    all_y,
    device,
    live_model,
    optimizer,
    loss_fn,
    shared_model,
    tf32=True,
    amp=True,
    mini_batch_size=2048,
    epochs=2,
):
    pin = device.type == "cuda"
    x_tensor = _concat_pinned(all_x, torch.float32, pin).to(device, non_blocking=True)
    a_tensor = (
        _concat_pinned(all_a, torch.int64, pin)
        .to(device, non_blocking=True)
        .unsqueeze(1)
    )
    y_tensor = _concat_pinned(all_y, torch.float32, pin).to(device, non_blocking=True)

    dataset_size = x_tensor.size(0)
    live_model.train()
    loss_sum = torch.zeros((), device=device)
    n_steps = 0
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if amp
        else contextlib.nullcontext()
    )

    with _tf32(tf32):
        for _ in range(epochs):
            perm = torch.randperm(dataset_size, device=device)
            x_shuf, a_shuf, y_shuf = x_tensor[perm], a_tensor[perm], y_tensor[perm]

            for i in range(0, dataset_size, mini_batch_size):
                batch_x = x_shuf[i : i + mini_batch_size]
                batch_a = a_shuf[i : i + mini_batch_size]
                batch_y = y_shuf[i : i + mini_batch_size]

                optimizer.zero_grad(set_to_none=True)
                with autocast:
                    predictions = live_model(batch_x).gather(1, batch_a)
                loss = loss_fn(predictions.float(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(live_model.parameters(), GRADIENT_CLIP)
                optimizer.step()
                loss_sum += loss.detach()
                n_steps += 1

    shared_model.load_state_dict(live_model.state_dict())

    return (loss_sum / max(n_steps, 1)).item()


@torch.no_grad()
def update_ema(ema_model, live_model, decay):
    for ema_p, live_p in zip(ema_model.parameters(), live_model.parameters()):
        ema_p.lerp_(live_p, 1.0 - decay)


def save_snapshot(
    snapshot_dir,
    batch_idx,
    live_model,
    ema_model,
    optimizer,
    resume_path,
    milestone: int | None = None,
    extra: dict | None = None,
):
    snapshot_path = os.path.join(snapshot_dir, f"model_gen_{batch_idx}.pt")
    # Everything that plays from a snapshot (evaluation, elites, stop checks,
    # benchmarks) reads "model_state_dict", so that holds the EMA weights. The
    # live weights are only needed to resume training.
    checkpoint = {
        "batch_idx": batch_idx,
        "model_state_dict": ema_model.state_dict(),
        "live_model_state_dict": live_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:  # e.g. a critic, which only training needs
        checkpoint.update(extra)
    torch.save(checkpoint, snapshot_path)
    torch.save(checkpoint, resume_path)

    if milestone is not None and batch_idx % milestone == 0:
        os.makedirs(os.path.join(snapshot_dir, "milestones"), exist_ok=True)
        milestone_path = os.path.join(
            snapshot_dir, f"milestones/model_gen_{batch_idx}.pt"
        )
        torch.save(checkpoint, milestone_path)
        print(f"Saved milestone snapshot to {milestone_path}.")

    return snapshot_path


def elite_subset(elite_paths, worker_id, k):
    """Balanced when NUM_WORKERS * k is a multiple of len(elite_paths)."""
    elites = sorted(elite_paths)
    if not k or len(elites) <= k:
        return elites
    return [elites[(worker_id * k + j) % len(elites)] for j in range(k)]


def shutdown(actors, stop_event, out_queue):
    stop_event.set()
    deadline = time.time() + 10

    while any(a.is_alive() for a in actors) and time.time() < deadline:
        try:
            while True:
                out_queue.get_nowait()
        except (queue.Empty, OSError, EOFError):
            pass
        for a in actors:
            a.join(timeout=0.1)
    for a in actors:
        if a.is_alive():
            a.kill()
