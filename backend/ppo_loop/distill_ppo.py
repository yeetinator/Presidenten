import os

# Avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import glob
import shutil
import time
import argparse
import numpy as np
import torch.multiprocessing as mp
from playerTypes.dmc_bot import PresidentValueNet, STATE_DIM
from playerTypes.ppo_bot import DMC_ENSEMBLE, PresidentPolicyNet, PresidentCriticNet
from utils.common import get_cached_model
from utils.ppo import (
    DISTILLED_PATH,
    DMC_ELITES_DIR,
    HELDOUT_PATH,
    PPO_DIR,
    actor_loop,
    build_batch,
    collect_games,
    critic_features,
    explained_variance,
    make_optimizer,
    masked_entropy,
    masked_log_softmax,
    predict_values,
)
from utils.training import shutdown

DMC_SNAPSHOT_DIR = "snapshots"

BATCH_GAMES = 60
HELDOUT_GAMES = 200
LOG_EVERY = 250
MINI_BATCH = 4096
LEARNING_RATE = 1e-4
CRITIC_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 12
GAMES_IN_FLIGHT_PER_WORKER = 64
ELITES_PER_WORKER = 2

ROLLOUT_TAU = 0.01  # DMC's final exploration temperature, for state coverage
TAU_CANDIDATES = (0.005, 0.01, 0.02, 0.05)
TARGET_ENTROPY = 0.25  # nats: PPO's starting exploration
MIN_MINUTES = 10
MAX_MINUTES = 60
PATIENCE = 3  # log windows without a 0.5% better held-out cross-entropy

INFERENCE_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def freeze_dmc_elites():
    """Copy DMC's elites once, so later DMC runs can't change PPO's anchors."""
    if glob.glob(f"{DMC_ELITES_DIR}/model_gen_*.pt"):
        return
    os.makedirs(DMC_ELITES_DIR, exist_ok=True)
    for path in glob.glob(f"{DMC_SNAPSHOT_DIR}/elites/model_gen_*.pt"):
        shutil.copy2(path, DMC_ELITES_DIR)
    print(f"Froze {len(os.listdir(DMC_ELITES_DIR))} DMC elites into {DMC_ELITES_DIR}")


def teacher_log_probs(teacher, batch, tau):
    with torch.no_grad():
        q = teacher(batch["x"])
    return q, masked_log_softmax(q / tau, batch["legal"])


def student_from_dmc(tau, device):
    """DMC.pt's weights with the head scaled by 1 / tau: a Boltzmann policy over
    one ensemble member, so cloning only has to move it to the ensemble."""
    student = PresidentPolicyNet().to(device)
    checkpoint = torch.load(DMC_ENSEMBLE[0], map_location=device)
    student.load_state_dict(checkpoint["model_state_dict"])
    with torch.no_grad():
        student.head.weight /= tau
        student.head.bias /= tau
    return student


@torch.no_grad()
def evaluate_heldout(student, critics, teacher_q, teacher_lp, batch):
    legal = batch["legal"]
    log_probs = masked_log_softmax(student(batch["x"]), legal)
    ce = -(teacher_lp.exp() * log_probs.masked_fill(~legal, 0.0)).sum(1).mean()

    q = teacher_q.masked_fill(~legal, float("-inf"))
    teacher_pick = q.argmax(1)
    student_pick = log_probs.argmax(1)
    rows = torch.arange(q.shape[0], device=q.device)
    outcome = batch["outcome"]
    return {
        "ce": ce.item(),
        "agree": (teacher_pick == student_pick).float().mean().item(),
        "regret": (q[rows, teacher_pick] - q[rows, student_pick]).mean().item(),
        "ent": masked_entropy(log_probs, legal).mean().item(),
        **{
            f"ev_{name}": explained_variance(
                predict_values(critic, name, batch), outcome
            )
            for name, (critic, _) in critics.items()
        },
        "ev_dmc": explained_variance(
            teacher_q.gather(1, batch["action"][:, None]).squeeze(1), outcome
        ),
    }


def choose_tau(teacher_q, batch):
    legal = batch["legal"]
    print(" Teacher entropy on held-out decisions by label temperature:")
    entropies = {}
    for tau in TAU_CANDIDATES:
        lp = masked_log_softmax(teacher_q / tau, legal)
        entropies[tau] = masked_entropy(lp, legal).mean().item()
        print(f"   tau {tau:<6} -> {entropies[tau]:.3f} nats")
    return min(TAU_CANDIDATES, key=lambda t: abs(entropies[t] - TARGET_ENTROPY))


def run_distillation(dagger=False, max_minutes=MAX_MINUTES, tau=None):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")  # the student's logits need fp32
    os.makedirs(PPO_DIR, exist_ok=True)
    freeze_dmc_elites()

    ctx = mp.get_context("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | DAgger: {dagger}")

    teacher = get_cached_model(DMC_ENSEMBLE, PresidentValueNet, device)
    shared_model = PresidentPolicyNet().to("cpu")
    shared_model.share_memory()
    if dagger:
        checkpoint = torch.load(DISTILLED_PATH, map_location=device)
        tau = checkpoint["tau_bc"]
        shared_model.load_state_dict(checkpoint["model_state_dict"])

    dmc_elites = glob.glob(f"{DMC_ELITES_DIR}/model_gen_*.pt")
    model_version = ctx.Value("i", 0, lock=False)
    stop_event = ctx.Event()
    out_queue = ctx.Queue(maxsize=2 * BATCH_GAMES)
    actors = [
        ctx.Process(
            target=actor_loop,
            args=(
                i,
                shared_model,
                model_version,
                INFERENCE_DEVICE,
                dmc_elites,
                [],
                stop_event,
                out_queue,
                GAMES_IN_FLIGHT_PER_WORKER,
                ELITES_PER_WORKER,
                None if dagger else ROLLOUT_TAU,
            ),
            daemon=True,
        )
        for i in range(NUM_WORKERS)
    ]
    for a in actors:
        a.start()

    try:
        if dagger:
            heldout = dict(np.load(HELDOUT_PATH))
        else:
            # Skip the first wave: short 4-player games finish first and would
            # be over-represented
            collect_games(out_queue, actors, NUM_WORKERS * GAMES_IN_FLIGHT_PER_WORKER)
            heldout_games, _, _ = collect_games(out_queue, actors, HELDOUT_GAMES)
            heldout = {
                key: np.concatenate([g[key] for g in heldout_games])
                for key in heldout_games[0]
                if key != "version"
            }
            np.savez_compressed(HELDOUT_PATH, **heldout)
        heldout_batch = build_batch([heldout], device)
        with torch.no_grad():
            heldout_q = teacher(heldout_batch["x"])
        print(f"Held-out set: {heldout_batch['x'].shape[0]} decisions")

        if tau is None:
            tau = choose_tau(heldout_q, heldout_batch)
        heldout_lp = masked_log_softmax(heldout_q / tau, heldout_batch["legal"])
        print(f"Label temperature tau_bc = {tau}")

        if dagger:
            student = PresidentPolicyNet().to(device)
            student.load_state_dict(checkpoint["model_state_dict"])
        else:
            student = student_from_dmc(tau, device)
        critics = {
            "full": PresidentCriticNet().to(device),
            "own": PresidentCriticNet(STATE_DIM).to(device),
        }
        if dagger:
            critics["full"].load_state_dict(checkpoint["critic_state_dict"])
            critics["own"].load_state_dict(checkpoint["own_critic_state_dict"])
        critics = {
            name: (net, make_optimizer(net, CRITIC_LEARNING_RATE, WEIGHT_DECAY))
            for name, net in critics.items()
        }
        optimizer = make_optimizer(student, LEARNING_RATE, WEIGHT_DECAY)

        print(
            "Starting distillation. Press Ctrl+C to stop and save the student.\n"
            + format_heldout(
                evaluate_heldout(student, critics, heldout_q, heldout_lp, heldout_batch)
            )
        )
        best_ce, stale = float("inf"), 0
        batch_idx = 0
        t_start = t_window = time.perf_counter()
        train_ces, rollout_times, train_times = [], [], []

        while True:
            t0 = time.perf_counter()
            games, _, _ = collect_games(out_queue, actors, BATCH_GAMES)
            t1 = time.perf_counter()
            batch = build_batch(games, device)
            _, target_lp = teacher_log_probs(teacher, batch, tau)
            train_ces.append(train_step(student, optimizer, critics, batch, target_lp))
            if dagger:
                shared_model.load_state_dict(student.state_dict())
                model_version.value += 1
            batch_idx += 1
            rollout_times.append(t1 - t0)
            train_times.append(time.perf_counter() - t1)

            if batch_idx % LOG_EVERY:
                continue
            t_now = time.perf_counter()
            stats = evaluate_heldout(
                student, critics, heldout_q, heldout_lp, heldout_batch
            )
            print(
                "=========================================================================\n"
                f"b {batch_idx} | ce {np.mean(train_ces):.4f} | "
                f"roll {np.mean(rollout_times):.2f}s train {np.mean(train_times):.2f}s | "
                f"{(t_now - t_window) / LOG_EVERY:.2f}s/it\n{format_heldout(stats)}"
            )
            t_window = t_now
            train_ces.clear()
            rollout_times.clear()
            train_times.clear()
            save_distilled(student, critics, tau, batch_idx)

            if stats["ce"] < best_ce * 0.995:
                best_ce, stale = stats["ce"], 0
            else:
                stale += 1
            minutes = (t_now - t_start) / 60
            if minutes >= max_minutes or (minutes >= MIN_MINUTES and stale >= PATIENCE):
                print(
                    f"Stopping after {minutes:.1f} min (plateau: {stale >= PATIENCE})."
                )
                break
    except KeyboardInterrupt:
        print("Distillation interrupted.")
    finally:
        shutdown(actors, stop_event, out_queue)


def train_step(student, optimizer, critics, batch, target_lp):
    """One pass over a batch: cross-entropy to the teacher for the student, the
    round outcome for both critics."""
    n = batch["x"].shape[0]
    perm = torch.randperm(n, device=batch["x"].device)
    ce_sum, n_steps = 0.0, 0
    student.train()
    for i in range(0, n, MINI_BATCH):
        rows = perm[i : i + MINI_BATCH]
        legal = batch["legal"][rows]
        log_probs = masked_log_softmax(student(batch["x"][rows]), legal)
        target = target_lp[rows].exp()
        loss = -(target * log_probs.masked_fill(~legal, 0.0)).sum(1).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        ce_sum += loss.item()
        n_steps += 1

    for name, (critic, critic_opt) in critics.items():
        train_critic_once(critic, critic_opt, name, batch, perm)
    return ce_sum / max(n_steps, 1)


def train_critic_once(critic, optimizer, name, batch, perm):
    critic.train()
    targets = batch["outcome"]
    device_type = targets.device.type
    for i in range(0, perm.numel(), MINI_BATCH):
        rows = perm[i : i + MINI_BATCH]
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            pred = critic(critic_features(name, batch, rows))
        loss = torch.nn.functional.mse_loss(pred.float(), targets[rows])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optimizer.step()


def format_heldout(s):
    return (
        f"   held-out | ce {s['ce']:.4f} agree {100 * s['agree']:.2f}% "
        f"regret {s['regret']:.5f} ent {s['ent']:.3f} | "
        f"ev full {s['ev_full']:.3f} own {s['ev_own']:.3f} dmc {s['ev_dmc']:.3f}"
    )


def save_distilled(student, critics, tau, batch_idx):
    torch.save(
        {
            "batch_idx": 0,  # PPO counts its iterations from here
            "distill_batches": batch_idx,
            "model_state_dict": student.state_dict(),
            "critic_state_dict": critics["full"][0].state_dict(),
            "own_critic_state_dict": critics["own"][0].state_dict(),
            "tau_bc": tau,
        },
        DISTILLED_PATH,
    )


def main():
    # python -m ppo_loop.distill_ppo [--dagger] [--minutes N] [--tau T]
    parser = argparse.ArgumentParser()
    parser.add_argument("--dagger", action="store_true")
    parser.add_argument("--minutes", type=float, default=MAX_MINUTES)
    parser.add_argument("--tau", type=float, default=None)
    args = parser.parse_args()
    run_distillation(args.dagger, args.minutes, args.tau)


if __name__ == "__main__":
    main()
