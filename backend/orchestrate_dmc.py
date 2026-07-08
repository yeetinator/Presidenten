import time
import subprocess
import os
import json
import glob
import shutil
import torch


def run_step(script_name, args=None):
    print(f"\n================ LAUNCHING {script_name.upper()} ================")
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(backend_dir, script_name)

    cmd = ["python", script_path]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, capture_output=False, text=True, cwd=backend_dir)
    if result.returncode != 0:
        print(f"Error: {script_name} failed with return code {result.returncode}.")
        return False
    return True


def manage_league_files(snapshot_dir="snapshots"):
    print("\n================ MANAGING LEAGUE POOL ================")

    json_path = f"{snapshot_dir}/evaluation_results.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Skipping league management.")
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    if not results:
        print("Error: No evaluation results found. Skipping league management.")
        return

    os.makedirs(f"{snapshot_dir}/elites", exist_ok=True)
    temp_dir = f"{snapshot_dir}/tmp"
    os.makedirs(temp_dir, exist_ok=True)

    def find_path(batch_num):
        possible_paths = [
            f"{snapshot_dir}/model_gen_{batch_num}.pt",
            f"{snapshot_dir}/elites/model_gen_{batch_num}.pt",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    top_8 = results[:8]
    print("Staging top 8 models")
    elite_count = 1

    for res in top_8:
        batch = res["batch"]
        src_path = find_path(batch)

        if src_path:
            shutil.copy2(src_path, f"{temp_dir}/elite_model_{batch}.pt")
            print(
                f"  -> Rank {elite_count}: Batch {batch} staged (Score: {res['avg_norm_score']:.4f})"
            )
            elite_count += 1

    print(f"Purging old files")
    for old_file in glob.glob(f"{snapshot_dir}/elites/model_gen_*.pt"):
        os.remove(old_file)

    for snap in glob.glob(f"{snapshot_dir}/model_gen_*.pt"):
        os.remove(snap)

    print("Committing staged files")
    for staged_elite in glob.glob(f"{temp_dir}/elite_model_*.pt"):
        batch = staged_elite.split("_")[-1].split(".")[0]
        shutil.move(staged_elite, f"{snapshot_dir}/elites/model_gen_{batch}.pt")

    shutil.rmtree(temp_dir)

    try:
        os.remove(json_path)
    except Exception as e:
        print(f"Error removing {json_path}: {e}")
    print("League management completed.")


def get_resume_cycle(snapshot_dir="snapshots"):
    resume_path = f"{snapshot_dir}/latest_model.pt"
    if os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=torch.device("cpu"))
            batch_idx = checkpoint.get("batch_idx", 0)
            return (batch_idx // 2000) + 1
        except Exception as e:
            print(f"Error loading checkpoint from {resume_path}: {e}")
    return 1


def run_orchestrator(
    snapshot_dir="snapshots",
    train_script="train_dmc.py",
    evaluate_script="evaluate_dmc.py",
):
    gen_cycle = get_resume_cycle(snapshot_dir)
    while True:
        print(f"\n=== STARTING LEAGUE GENERATION CYCLE {gen_cycle} ===")
        if not run_step(train_script):
            break
        if not run_step(evaluate_script, [str(gen_cycle)]):
            break

        manage_league_files(snapshot_dir)
        gen_cycle += 1
        time.sleep(5)


def main():
    run_orchestrator(
        snapshot_dir="snapshots",
        train_script="train_dmc.py",
        evaluate_script="evaluate_dmc.py",
    )


if __name__ == "__main__":
    main()
