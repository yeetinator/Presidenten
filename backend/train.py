import torch
import glob
import random
import os
import numpy as np
from playerTypes.dmc_bot import PresidentenDMCBot, PresidentenValueNet
from game import Presidenten

BATCH_GAMES = 50
ROUNDS_PER_GAME = 10
SAVE_SNAPSHOT_EVERY = 20
LEARNING_RATE = 1e-4
INPUT_DIM = 91


def run_single_game(live_model, device, league_models=None):
    bot_instances: dict[int, PresidentenDMCBot] = {}
    use_snapshot = league_models is not None and random.random() < 0.2
    snapshot_seats = random.sample(range(4), k=2) if use_snapshot else []

    for seat in range(4):
        if seat in snapshot_seats and league_models:
            snap_model = random.choice(league_models)
            bot_instances[seat] = PresidentenDMCBot(
                player_id=seat, model=snap_model, device=device, training=False
            )
        else:
            bot_instances[seat] = PresidentenDMCBot(
                player_id=seat, model=live_model, device=device, training=True
            )
            bot_instances[seat].trajectory = []

    env = Presidenten(players=4)
    for round_idx in range(ROUNDS_PER_GAME):
        state = env.full_reset(next_round=(round_idx > 0))
        if round_idx > 0:
            cards_to_pass = {}
            for p_id, role in env.roles.items():
                if role != "Citizen":
                    cards_to_pass[p_id] = bot_instances[p_id].choose_cards_to_pass(
                        env._get_state(p_id)
                    )

            env.exchange_cards(cards_to_pass)
            state = env._get_state(env.curr_turn)

        while not env.game_over:
            curr_player = env.curr_turn
            if curr_player is None:
                break

            chosen_move = bot_instances[curr_player].get_move(state, env)
            state, _ = env.step(curr_player, chosen_move)
        env.assign_roles()

    game_x, game_y = [], []
    max_possible_score = env.players - 1

    for p_id, bot in bot_instances.items():
        if bot.training and len(bot.trajectory) > 0:
            raw_score = env.scores[p_id]
            normalized_score = (
                raw_score[0] / (max_possible_score * ROUNDS_PER_GAME)
            ) * 2 - 1

            for features in bot.trajectory:
                game_x.append(features)
                game_y.append([normalized_score])
            bot.trajectory.clear()
    return game_x, game_y


def main():
    os.makedirs("snapshots", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    live_model = PresidentenValueNet(INPUT_DIM).to(device)
    optimizer = torch.optim.Adam(live_model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    batch_idx = 0
    print("Starting training loop. Press Ctrl+C to stop and save the model.")

    try:
        while True:
            batch_idx += 1
            all_x, all_y = [], []
            snapshot_files = glob.glob("snapshots/model_gen_*.pt")
            loaded_league_models = None

            if snapshot_files:
                selected_files = random.sample(
                    snapshot_files, k=min(3, len(snapshot_files))
                )
                loaded_league_models = []

                for snap_file in selected_files:
                    model = PresidentenValueNet(INPUT_DIM).to(device)
                    model.load_state_dict(torch.load(snap_file, map_location=device))
                    model.eval()
                    loaded_league_models.append(model)

            live_model.eval()
            for _ in range(BATCH_GAMES):
                game_x, game_y = run_single_game(
                    live_model, device, loaded_league_models
                )
                all_x.extend(game_x)
                all_y.extend(game_y)

            if all_x:
                x_tensor = torch.FloatTensor(np.array(all_x)).to(device)
                y_tensor = torch.FloatTensor(np.array(all_y)).to(device)

                live_model.train()
                optimizer.zero_grad()

                predictions = live_model(x_tensor)
                loss = loss_fn(predictions, y_tensor)

                loss.backward()
                optimizer.step()
                print(
                    f"Batch {batch_idx}: Loss = {loss.item():.6f}, Games = {len(all_x)}"
                )
            else:
                print(f"Batch {batch_idx}: No training data collected.")

            if batch_idx % SAVE_SNAPSHOT_EVERY == 0:
                snapshot_path = f"snapshots/model_gen_{batch_idx}.pt"
                torch.save(live_model.state_dict(), snapshot_path)
                print(f"Saved model snapshot to {snapshot_path}")
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        snapshot_path = f"snapshots/model_gen_{batch_idx}.pt"
        torch.save(live_model.state_dict(), snapshot_path)
        print(f"Saved model snapshot to {snapshot_path}")


if __name__ == "__main__":
    main()
