import time
import subprocess
import os
import json
import glob
import shutil


def run_step(script_name):
    print(f"\n================ LAUNCHING {script_name.upper()} ================")
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(backend_dir, script_name)

    result = subprocess.run(
        ["python", script_path], capture_output=False, text=True, cwd=backend_dir
    )
    if result.returncode != 0:
        print(f"Error: {script_name} failed with return code {result.returncode}.")
        return False
    return True


def manage_league_files():
    print("\n================ MANAGING LEAGUE POOL ================")

    json_path = "snapshots/evaluation_results.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Skipping league management.")
        return

    with open(json_path, "r") as f:
        results = json.load(f)

    if not results:
        print("Error: No evaluation results found. Skipping league management.")
        return

    os.makedirs("snapshots/elites", exist_ok=True)
    os.makedirs("snapshots/graduated", exist_ok=True)
    temp_dir = "snapshots/tmp"
    os.makedirs(temp_dir, exist_ok=True)

    def find_path(batch_num):
        possible_paths = [
            f"snapshots/model_gen_{batch_num}.pt",
            f"snapshots/elites/model_gen_{batch_num}.pt",
            f"snapshots/graduated/model_gen_{batch_num}.pt",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    graduated_empty = len(glob.glob("snapshots/graduated/model_gen_*.pt")) == 0
    graduated_batches = set()

    if graduated_empty:
        valid_candidates = [r for r in results if r["avg_norm_score"] > 0.35]
        valid_candidates.sort(key=lambda x: (x["avg_norm_score"], x["wins"]))
        worst_2 = valid_candidates[:2]

        print(f"Graduated folder is empty. Moving {len(worst_2)} anchor models:")
        for res in worst_2:
            batch = res["batch"]
            src_path = find_path(batch)

            if src_path:
                shutil.copy2(src_path, f"{temp_dir}/grad_model_{batch}.pt")
                graduated_batches.add(batch)
                print(
                    f"  -> Staged graduated: Batch {batch} (Score: {res['avg_norm_score']:.4f})"
                )

    top_8 = results[:8]
    print("Staging top 8 models")
    elite_count = 1

    for res in top_8:
        batch = res["batch"]
        if batch in graduated_batches:
            continue

        src_path = find_path(batch)
        if src_path:
            shutil.copy2(src_path, f"{temp_dir}/elite_model_{batch}.pt")
            print(
                f"  -> Rank {elite_count}: Batch {batch} staged (Score: {res['avg_norm_score']:.4f})"
            )
            elite_count += 1

    print(f"Purging old files")
    for old_file in glob.glob("snapshots/elites/model_gen_*.pt"):
        os.remove(old_file)

    for snap in glob.glob("snapshots/model_gen_*.pt"):
        os.remove(snap)

    print("Committing staged files")
    for staged_grad in glob.glob(f"{temp_dir}/grad_model_*.pt"):
        batch = staged_grad.split("_")[-1].split(".")[0]
        shutil.move(staged_grad, f"snapshots/graduated/model_gen_{batch}.pt")

    for staged_elite in glob.glob(f"{temp_dir}/elite_model_*.pt"):
        batch = staged_elite.split("_")[-1].split(".")[0]
        shutil.move(staged_elite, f"snapshots/elites/model_gen_{batch}.pt")

    shutil.rmtree(temp_dir)

    try:
        os.remove(json_path)
    except Exception as e:
        print(f"Error removing {json_path}: {e}")
    print("League management completed.")


def main():
    gen_cycle = 1
    while True:
        print(f"\n=== STARTING LEAGUE GENERATION CYCLE {gen_cycle} ===")
        if not run_step("train.py"):
            break
        if not run_step("evaluate.py"):
            break

        manage_league_files()
        gen_cycle += 1
        time.sleep(5)


if __name__ == "__main__":
    main()
