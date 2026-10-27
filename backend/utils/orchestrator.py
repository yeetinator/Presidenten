import os
import json
import torch
import shutil
import glob
import sys
import subprocess
from playerTypes.dmc_bot import PresidentValueNet, PresidentDMCBot
from .head_to_head import run_h2h, format_h2h

DMC_CLASSES = (PresidentValueNet, PresidentDMCBot)
BATCHES_PER_CYCLE = 2000
STOP_CHECK_EVERY = 3
STOP_CHECK_GAMES = 1200
STOP_PATIENCE = 2
STOP_PATIENCE_AFTER_CUT = 3
STOP_MIN_CYCLES = 15
LR_CUT_FACTOR = 3.0
LR_MIN_SCALE = 0.1  # Cuts that would go below this are skipped: 1 -> 1/3 -> 1/9


def _best_of_cycle(snapshot_dir, gen_cycle):
    json_path = f"{snapshot_dir}/evals/evaluation_results_{gen_cycle}.json"
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        results = json.load(f)
    return results[0] if results else None


def save_benchmark_ref(snapshot_dir, gen_cycle):
    best = _best_of_cycle(snapshot_dir, gen_cycle)
    if best is None:
        return None
    src = f"{snapshot_dir}/elites/model_gen_{best['batch']}.pt"
    if not os.path.exists(src):
        return None

    os.makedirs(f"{snapshot_dir}/bench_refs", exist_ok=True)
    dst = f"{snapshot_dir}/bench_refs/cycle_{gen_cycle}.pt"
    checkpoint = torch.load(src, map_location="cpu")
    torch.save(
        {
            "batch_idx": checkpoint["batch_idx"],
            "model_state_dict": checkpoint["model_state_dict"],
        },
        dst,
    )
    return dst, best["batch"]


def _load_stop_check(snapshot_dir, gen_cycle):
    path = f"{snapshot_dir}/evals/stop_check_{gen_cycle}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def current_lr_scale(snapshot_dir):
    """LR multiplier set by the latest stop check, 1.0 before any cut."""
    paths = glob.glob(f"{snapshot_dir}/evals/stop_check_*.json")
    if not paths:
        return 1.0
    latest = max(paths, key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    with open(latest) as f:
        return json.load(f).get("lr_scale", 1.0)


def _anchor_after(snapshot_dir, cycle, check_every):
    # The anchor is the last cycle that significantly beat its own anchor. A
    # check cycle without a record (the first one) starts as the anchor.
    while cycle > 0:
        record = _load_stop_check(snapshot_dir, cycle)
        if record is None or record["improved"]:
            return cycle
        if "anchor_cycle" in record:
            return record["anchor_cycle"]
        # Older records compared against the previous check, so keep walking
        cycle -= check_every
    return None


def should_stop_training(
    snapshot_dir,
    gen_cycle,
    check_every=STOP_CHECK_EVERY,
    games=STOP_CHECK_GAMES,
    min_cycles=STOP_MIN_CYCLES,
    classes=DMC_CLASSES,
):
    if gen_cycle % check_every != 0:
        return False

    # Compare against a fixed anchor rather than the previous check: per-check
    # gains can be too small to detect in one H2H while still adding up
    cur = save_benchmark_ref(snapshot_dir, gen_cycle)
    prev_cycle = gen_cycle - check_every
    anchor_cycle = _anchor_after(snapshot_dir, prev_cycle, check_every)
    anchor_ref = f"{snapshot_dir}/bench_refs/cycle_{anchor_cycle}.pt"
    anchor_best = _best_of_cycle(snapshot_dir, anchor_cycle) if anchor_cycle else None

    if cur is None or anchor_best is None or not os.path.exists(anchor_ref):
        print(f"  [stop check] Cycle {gen_cycle}: no anchor yet, skipped.")
        return False

    cur_ref, cur_batch = cur
    anchor_batch = anchor_best["batch"]

    if cur_batch == anchor_batch:
        summary = None
        improved = False
        report = (
            f" Best elite unchanged since anchor cycle {anchor_cycle} "
            f"(batch {cur_batch})."
        )
    else:
        print(
            f"  [stop check] Cycle {gen_cycle}: batch {cur_batch} vs anchor "
            f"cycle {anchor_cycle} (batch {anchor_batch}, {games} games)..."
        )
        summary = run_h2h(
            cur_ref,
            anchor_ref,
            games,
            master_classes=classes,
            old_classes=classes,
            progress_every=0,
        )
        improved = summary["ci_lo"] > 0
        report = format_h2h(summary)

    prev_record = _load_stop_check(snapshot_dir, prev_cycle)
    streak = 0 if improved else (prev_record["streak"] if prev_record else 0) + 1

    # A stall first lowers the LR, since smaller steps average out more of the
    # noise in the targets. Only stop once the LR has reached its floor.
    lr_scale = current_lr_scale(snapshot_dir)
    patience = STOP_PATIENCE if lr_scale >= 1.0 else STOP_PATIENCE_AFTER_CUT
    stalled = gen_cycle >= min_cycles and streak >= patience
    lr_cut = stalled and lr_scale / LR_CUT_FACTOR >= LR_MIN_SCALE

    if lr_cut:
        lr_scale /= LR_CUT_FACTOR
        streak = 0
        patience = STOP_PATIENCE_AFTER_CUT
    stop = stalled and not lr_cut

    action = f"LR CUT to {lr_scale:.3g}x" if lr_cut else "STOP" if stop else "continue"
    decision = (
        f" Cycle {gen_cycle}: batch {cur_batch} vs anchor cycle {anchor_cycle} "
        f"(batch {anchor_batch}) -> "
        f"{'IMPROVED, new anchor' if improved else 'no significant improvement'} | "
        f"streak {streak}/{patience} | {action}"
    )

    os.makedirs(f"{snapshot_dir}/evals", exist_ok=True)
    with open(f"{snapshot_dir}/evals/stop_check_{gen_cycle}.json", "w") as f:
        json.dump(
            {
                "cycle": gen_cycle,
                "current_batch": cur_batch,
                "anchor_cycle": anchor_cycle,
                "reference_batch": anchor_batch,
                "improved": improved,
                "streak": streak,
                "lr_scale": lr_scale,
                "lr_cut": lr_cut,
                "stop": stop,
                "summary": summary,
            },
            f,
            indent=4,
        )
    with open(f"{snapshot_dir}/evals/stop_checks.log", "a") as f:
        f.write(f"\n### STOP CHECK, CYCLE {gen_cycle} ###\n{report}\n{decision}\n")

    print(report)
    print(f"  [stop check]{decision}")
    return stop


def manage_league_files(snapshot_dir, gen_cycle=None):
    print("\n================ MANAGING LEAGUE POOL ================")

    json_path = f"{snapshot_dir}/evals/evaluation_results_{gen_cycle}.json"
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

    top_8 = results[:8]
    print("Staging top 8 models")

    for rank, res in enumerate(top_8, start=1):
        batch = res["batch"]
        src_path = res["path"]

        if os.path.exists(src_path):
            shutil.copy2(src_path, f"{temp_dir}/elite_model_{batch}.pt")
            print(f"  -> Rank {rank}: Batch {batch:<5} staged (Elo: {res['elo']:.2f})")

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
    print("League management completed.")


def run_step(module_name, args=None, dir_name=None):
    print(f"\n================ LAUNCHING {module_name.upper()} ================")
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_name = dir_name if dir_name else os.path.basename(backend_dir)
    module_path = f"{folder_name}.{module_name}"

    cmd = [sys.executable, "-m", module_path]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, capture_output=False, text=True, cwd=backend_dir)
    if result.returncode != 0:
        print(f"Error: {module_name} failed with return code {result.returncode}.")
        return False
    return True


def get_resume_cycle(snapshot_dir, batches_per_cycle=BATCHES_PER_CYCLE):
    resume_path = f"{snapshot_dir}/latest_model.pt"
    if os.path.exists(resume_path):
        try:
            checkpoint = torch.load(resume_path, map_location=torch.device("cpu"))
            batch_idx = checkpoint.get("batch_idx", 0)
            return (batch_idx // batches_per_cycle) + 1
        except Exception as e:
            print(f"Error loading checkpoint from {resume_path}: {e}")
    return 1


def run_orchestrator(
    snapshot_dir,
    train_script,
    evaluate_script,
    dir_name,
    classes=DMC_CLASSES,
    batches_per_cycle=BATCHES_PER_CYCLE,
    after_cycle=None,
):
    gen_cycle = get_resume_cycle(snapshot_dir, batches_per_cycle)
    while True:
        print(f"\n=== STARTING LEAGUE GENERATION CYCLE {gen_cycle} ===")
        if not run_step(train_script, dir_name=dir_name):
            break
        if not run_step(evaluate_script, [str(gen_cycle)], dir_name):
            break

        manage_league_files(snapshot_dir, gen_cycle)

        stop = should_stop_training(snapshot_dir, gen_cycle, classes=classes)
        if stop:
            print("Training has plateaued.")

        # Runs after the stop check, which saves this cycle's benchmark ref
        if after_cycle is not None and after_cycle(snapshot_dir, gen_cycle):
            stop = True
        if stop:
            break

        gen_cycle += 1
