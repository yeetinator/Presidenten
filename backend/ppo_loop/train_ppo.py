import os
import random
import glob
import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing as mp
from game import President
from playerTypes.player import Player
from playerTypes.ppo_bot import (
    PresidentActorCritic,
    PresidentPPOBot,
    MOVE_TO_IDX,
    IDX_TO_MOVE,
    ACTION_DIM,
    get_action_mask,
)
from playerTypes.baseline_bot import PresidentBaselineBot
from playerTypes.dmc_bot import (
    vectorize_state,
    vectorize_master_state,
    PresidentValueNet,
    PresidentDMCBot,
)
from utils import (
    init_worker,
    get_cached_model,
    prune_cache,
    save_snapshot,
    handle_keyboard_interrupt,
)

GAMMA = 1.0
GAE_LAMBDA = 0.95
PPO_EPOCHS = 4
BATCH_SIZE = 512
CLIP_EPS = 0.2
LR = 3e-4
SAVE_SNAPSHOT_EVERY = 250
NUM_WORKERS = 12
BATCH_GAMES = 48


def parallel_worker(shared_model, elite_snapshots, num_players, basic_elites):
    active_paths = set(basic_elites + elite_snapshots)

    for p in active_paths:
        net_cls = PresidentActorCritic if p in elite_snapshots else PresidentValueNet
        get_cached_model(p, net_cls)

    prune_cache(active_paths)
    return collect_single_game_rollout(
        shared_model, elite_snapshots, num_players, basic_elites
    )


def collect_single_game_rollout(
    shared_model, elite_snapshots, num_players, basic_elites
):
    device = torch.device("cpu")
    env = President(num_players)

    ppo_seats = {0}
    bot_instances: dict[int, Player] = {0: PresidentPPOBot(0, shared_model, device)}

    for seat in range(1, env.players):
        roll = random.random()
        if roll < 0.40:
            ppo_seats.add(seat)
            bot_instances[seat] = PresidentPPOBot(seat, shared_model, device)
        elif roll < 0.60 and elite_snapshots:
            snap_path = random.choice(elite_snapshots)
            snap_model = get_cached_model(snap_path, PresidentActorCritic)
            bot_instances[seat] = PresidentPPOBot(seat, snap_model, device)
        elif roll < 0.75 and basic_elites:
            snap_path = random.choice(basic_elites)
            model = get_cached_model(snap_path, PresidentValueNet)
            bot_instances[seat] = PresidentDMCBot(seat, model, device)
        elif roll < 0.90 and basic_elites:
            snap_path = random.choice(basic_elites)
            model = get_cached_model(snap_path, PresidentValueNet)
            bot_instances[seat] = PresidentDMCBot(
                seat, model, device, profile="aggressive"
            )
        else:
            bot_instances[seat] = PresidentBaselineBot(seat)

    trajectories = {
        seat: {
            "obs": [],
            "priv": [],
            "actions": [],
            "log_probs": [],
            "masks": [],
            "rewards": [],
        }
        for seat in ppo_seats
    }

    for round_idx in range(10):
        state = env.full_reset(round_idx > 0)
        if round_idx > 0:
            cards_to_pass = {}
            for p_id, role in env.roles.items():
                if role != "Citizen":
                    cards_to_pass[p_id] = bot_instances[p_id].choose_cards_to_pass(
                        env._get_state(p_id)
                    )

            for pair in env.role_pairs:
                env.exchange_cards(pair, cards_to_pass)
            state = env._get_state(env.curr_turn)

        while not env.game_over:
            curr_player = env.curr_turn
            if curr_player is None:
                break

            mask_np = get_action_mask(state["legal_moves"])
            if curr_player in ppo_seats:
                obs_v = vectorize_state(state, env.players)
                priv_v = vectorize_master_state(state, env, env.players)
                obs_t = torch.FloatTensor(obs_v).unsqueeze(0).to(device)
                mask_t = torch.BoolTensor(mask_np).unsqueeze(0).to(device)

                with torch.no_grad():
                    dist = shared_model.forward_actor(obs_t, mask_t)
                    action_tensor = dist.sample()
                    log_prob = dist.log_prob(action_tensor)

                action_idx = int(action_tensor.item())
                move = IDX_TO_MOVE[action_idx]

                trajectories[curr_player]["obs"].append(obs_v)
                trajectories[curr_player]["priv"].append(priv_v)
                trajectories[curr_player]["actions"].append(action_idx)
                trajectories[curr_player]["log_probs"].append(log_prob.item())
                trajectories[curr_player]["masks"].append(mask_np)
                trajectories[curr_player]["rewards"].append(0.0)
            else:
                move = bot_instances[curr_player].get_move(state, env)
            state, _ = env.step(curr_player, move)

            if env.was_pile_reset:
                env.clear_pile()

        env.assign_roles()
        max_pos_score = env.players - 1

        for rank, p_id in enumerate(env.out_order):
            if p_id in ppo_seats and len(trajectories[p_id]["rewards"]) > 0:
                normalized_score = ((max_pos_score - rank) / max_pos_score) * 2 - 1
                trajectories[p_id]["rewards"][-1] = normalized_score
    return trajectories


def train_ppo_step(model, optimizer, scheduler, batch_data, device):
    obs = torch.tensor(np.array(batch_data["obs"]), dtype=torch.float32).to(device)
    priv = torch.tensor(np.array(batch_data["priv"]), dtype=torch.float32).to(device)
    actions = torch.tensor(batch_data["actions"], dtype=torch.long).to(device)
    old_log_probs = torch.tensor(batch_data["log_probs"], dtype=torch.float32).to(
        device
    )
    masks = torch.tensor(np.array(batch_data["masks"]), dtype=torch.bool).to(device)
    returns = torch.tensor(batch_data["returns"], dtype=torch.float32).to(device)
    advantages = torch.tensor(batch_data["advantages"], dtype=torch.float32).to(device)

    dataset_size = obs.size(0)
    step_losses = []

    model.train()
    for _ in range(PPO_EPOCHS):
        perm = torch.randperm(dataset_size)
        for i in range(0, dataset_size, BATCH_SIZE):
            indices = perm[i : i + BATCH_SIZE]

            b_obs = obs[indices]
            b_priv = priv[indices]
            b_actions = actions[indices]
            b_old_log_probs = old_log_probs[indices]
            b_masks = masks[indices]
            b_returns = returns[indices]
            b_advantages = advantages[indices]

            dist = model.forward_actor(b_obs, b_masks)
            v_preds = model.forward_critic(b_priv).squeeze(-1)

            new_log_probs = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - b_old_log_probs)
            surr1 = ratios * b_advantages
            surr2 = torch.clamp(ratios, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(v_preds, b_returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            step_losses.append(loss.item())

    scheduler.step()
    return np.mean(step_losses)


def run_training_loop(snapshot_dir="snapshots_ppo", basic_snapshot_dir="snapshots"):
    os.makedirs(snapshot_dir, exist_ok=True)

    ctx = mp.get_context("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    resume_path = os.path.join(snapshot_dir, "latest_model.pt")
    live_model = PresidentActorCritic().to(device)
    shared_model = PresidentActorCritic().to("cpu")
    shared_model.share_memory()

    optimizer = torch.optim.Adam(live_model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99995)

    if os.path.exists(resume_path):
        print(f"Resuming PPO checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        live_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        batch_idx = checkpoint["batch_idx"]
    else:
        batch_idx = 0

    print("Beginning PPO execution loop")
    max_batches = 0 if batch_idx == 0 else batch_idx % 2000
    running_losses = []
    elite_snapshots = glob.glob(os.path.join(snapshot_dir, "elites/model_gen_*.pt"))
    basic_elites = glob.glob(
        os.path.join(basic_snapshot_dir, "model_gen_*.pt")
    ) + glob.glob(os.path.join(basic_snapshot_dir, "elites/model_gen_*.pt"))

    with ctx.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        try:
            while max_batches < 2000:
                batch_idx += 1
                max_batches += 1

                shared_model.load_state_dict(live_model.state_dict())
                num_players = 4 + (batch_idx % 4)

                tasks = [
                    (shared_model, elite_snapshots, num_players, basic_elites)
                    for _ in range(BATCH_GAMES)
                ]
                try:
                    results = pool.starmap_async(parallel_worker, tasks)
                    game_trajectories = results.get(timeout=600)
                except mp.TimeoutError:
                    print(f"\nBatch {batch_idx} timed out.")
                    continue

                global_batch = {
                    "obs": [],
                    "priv": [],
                    "actions": [],
                    "log_probs": [],
                    "masks": [],
                    "returns": [],
                    "advantages": [],
                }
                live_model.eval()

                for trajectories in game_trajectories:
                    for seat, traj in trajectories.items():
                        if not traj["obs"]:
                            continue

                        priv_t = torch.FloatTensor(np.array(traj["priv"])).to(device)
                        with torch.no_grad():
                            values = (
                                live_model.forward_critic(priv_t)
                                .squeeze(-1)
                                .cpu()
                                .numpy()
                            )

                        values = np.append(values, 0.0)
                        rewards = np.array(traj["rewards"])
                        advantages = np.zeros_like(rewards)
                        last_gae_lam = 0

                        for t in reversed(range(len(rewards))):
                            delta = rewards[t] + GAMMA * values[t + 1] - values[t]
                            advantages[t] = last_gae_lam = (
                                delta + GAMMA * GAE_LAMBDA * last_gae_lam
                            )
                        returns = advantages + values[:-1]

                        global_batch["obs"].extend(traj["obs"])
                        global_batch["priv"].extend(traj["priv"])
                        global_batch["actions"].extend(traj["actions"])
                        global_batch["log_probs"].extend(traj["log_probs"])
                        global_batch["masks"].extend(traj["masks"])
                        global_batch["returns"].extend(returns.tolist())
                        global_batch["advantages"].extend(advantages.tolist())

                if len(global_batch["obs"]) >= BATCH_SIZE:
                    adv_np = np.array(global_batch["advantages"])
                    global_batch["advantages"] = (
                        (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)
                    ).tolist()

                    avg_loss = train_ppo_step(
                        live_model, optimizer, scheduler, global_batch, device
                    )
                    running_losses.append(avg_loss)

                    if batch_idx % 25 == 0:
                        rolling_loss = np.mean(running_losses)
                        print(
                            f"=========================================================================\n"
                            f"PPO BATCH {batch_idx:<5} | Loss: {rolling_loss:.6f} | "
                            f"LR: {scheduler.get_last_lr()[0]:.6f}\n"
                            f"========================================================================="
                        )
                        running_losses.clear()
                if batch_idx % SAVE_SNAPSHOT_EVERY == 0:
                    snapshot_path = save_snapshot(
                        snapshot_dir,
                        batch_idx,
                        live_model,
                        optimizer,
                        scheduler,
                        None,
                        resume_path,
                    )
                    print(
                        f"Saved PPO snapshot to {snapshot_path} and updated latest_model.pt"
                    )
        except KeyboardInterrupt:
            handle_keyboard_interrupt(pool)


def main():
    run_training_loop()


if __name__ == "__main__":
    main()
