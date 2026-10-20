import torch
import os
import contextlib
import numpy as np

GRADIENT_CLIP = 1.0


def train_on_batch(
    all_x, all_a, all_y, device, live_model, optimizer, loss_fn, shared_model, scheduler
):
    merged_x = np.concatenate(all_x, axis=0)
    merged_a = np.concatenate(all_a, axis=0)
    merged_y = np.concatenate(all_y, axis=0)

    x_tensor = torch.from_numpy(merged_x).to(device)
    a_tensor = torch.from_numpy(merged_a).to(device).unsqueeze(1)
    y_tensor = torch.from_numpy(merged_y).to(device)

    dataset_size = x_tensor.size(0)
    mini_batch_size = 2048
    epochs = 2

    live_model.train()
    loss_sum = torch.zeros((), device=device)
    n_steps = 0

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size, device=device)
        x_shuf, a_shuf, y_shuf = x_tensor[perm], a_tensor[perm], y_tensor[perm]

        for i in range(0, dataset_size, mini_batch_size):
            batch_x = x_shuf[i : i + mini_batch_size]
            batch_a = a_shuf[i : i + mini_batch_size]
            batch_y = y_shuf[i : i + mini_batch_size]

            optimizer.zero_grad(set_to_none=True)
            predictions = live_model(batch_x).gather(1, batch_a)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(live_model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            loss_sum += loss.detach()
            n_steps += 1

    shared_model.load_state_dict(live_model.state_dict())
    scheduler.step()

    return (loss_sum / max(n_steps, 1)).item()


def save_snapshot(
    snapshot_dir,
    batch_idx,
    live_model,
    optimizer,
    scheduler,
    resume_path,
    milestone: int | None = None,
):
    snapshot_path = os.path.join(snapshot_dir, f"model_gen_{batch_idx}.pt")
    checkpoint = {
        "batch_idx": batch_idx,
        "model_state_dict": live_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
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
