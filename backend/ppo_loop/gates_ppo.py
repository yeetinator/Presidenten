import os
import sys
import json
import functools
import torch
from playerTypes.ppo_bot import DMC_ENSEMBLE, PresidentPolicyNet, PresidentPPOBot
from utils.head_to_head import run_h2h, format_h2h
from utils.orchestrator import STOP_CHECK_EVERY
from utils.ppo import DISTILLED_PATH, PPO_CLASSES, PPO_DIR

GATE_GAMES = 2400
EMA_CHECK_GAMES = 1200
# Gate 1 passes unless the distilled actor is clearly worse than the ensemble.
# With n = 2400 the half-width is about 0.0065, so this admits a true deficit
# of at most about 0.01
GATE1_MIN_CI_LO = -0.012
GIVE_UP_CYCLE = 15

GREEDY = (PresidentPolicyNet, PresidentPPOBot)
SAMPLED = (PresidentPolicyNet, functools.partial(PresidentPPOBot, sample=True))


def _h2h(label, master, old, master_classes, old_classes=None, games=GATE_GAMES):
    print(f"\n  [gate] {label} ({games} games)...")
    kwargs = {"old_classes": old_classes} if old_classes else {}
    summary = run_h2h(
        master, old, games, master_classes=master_classes, progress_every=0, **kwargs
    )
    summary["label"] = label
    print(format_h2h(summary))
    return summary


def _log(snapshot_dir, name, record, lines):
    os.makedirs(f"{snapshot_dir}/evals", exist_ok=True)
    with open(f"{snapshot_dir}/evals/{name}.json", "w") as f:
        json.dump(record, f, indent=4, default=str)
    with open(f"{snapshot_dir}/evals/gates.log", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _line(s):
    return (
        f"  {s['label']:<34}: delta {s['delta']:+.4f} "
        f"CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] n={s['total_games']}"
    )


def run_gate1(snapshot_dir=PPO_DIR):
    """The distilled actor must play at parity with the DMC ensemble. Also picks
    the evaluation mode: greedy or sampled."""
    greedy = _h2h("distilled greedy vs ensemble", DISTILLED_PATH, DMC_ENSEMBLE, GREEDY)
    sampled = _h2h(
        "distilled sampled vs ensemble", DISTILLED_PATH, DMC_ENSEMBLE, SAMPLED
    )
    modes = _h2h(
        "distilled sampled vs greedy", DISTILLED_PATH, DISTILLED_PATH, SAMPLED, GREEDY
    )

    passed = greedy["ci_hi"] >= 0 and greedy["ci_lo"] >= GATE1_MIN_CI_LO
    better_mode = "sampled" if modes["ci_lo"] > 0 else "greedy"
    lines = [
        "\n### GATE 1: DISTILLED ACTOR vs DMC ENSEMBLE ###",
        *(_line(s) for s in (greedy, sampled, modes)),
        f"  -> {'PASS' if passed else 'FAIL'} (needs CI to reach 0 and "
        f"ci_lo >= {GATE1_MIN_CI_LO}) | evaluation mode: {better_mode}",
    ]
    _log(
        snapshot_dir,
        "gate1",
        {"passed": passed, "eval_mode": better_mode, "runs": [greedy, sampled, modes]},
        lines,
    )
    return passed


def _load_gate2(snapshot_dir, cycle):
    path = f"{snapshot_dir}/evals/gate2_{cycle}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _live_ref(snapshot_dir, gen_cycle):
    """The cycle's best elite with its live weights instead of the EMA ones."""
    with open(f"{snapshot_dir}/evals/evaluation_results_{gen_cycle}.json") as f:
        best = json.load(f)[0]
    checkpoint = torch.load(
        f"{snapshot_dir}/elites/model_gen_{best['batch']}.pt", map_location="cpu"
    )
    dst = f"{snapshot_dir}/bench_refs/cycle_{gen_cycle}_live.pt"
    torch.save({"model_state_dict": checkpoint["live_model_state_dict"]}, dst)
    return dst


def check_gate2(snapshot_dir, gen_cycle, check_every=STOP_CHECK_EVERY):
    """Gate 2 at every stop-check cycle: the best elite against the ensemble and
    the distilled start. Returns True when the run should give up."""
    if gen_cycle % check_every != 0:
        return False
    ref = f"{snapshot_dir}/bench_refs/cycle_{gen_cycle}.pt"
    if not os.path.exists(ref):
        print(f"  [gate] Cycle {gen_cycle}: no benchmark ref, skipped.")
        return False

    vs_ensemble = _h2h("best elite vs ensemble", ref, DMC_ENSEMBLE, PPO_CLASSES)
    vs_distilled = _h2h(
        "best elite vs distilled", ref, DISTILLED_PATH, PPO_CLASSES, PPO_CLASSES
    )
    runs = [vs_ensemble, vs_distilled]

    # Testing at every check inflates false passes, so a pass against the
    # ensemble has to repeat on fresh deals
    beats_ensemble = vs_ensemble["ci_lo"] > 0
    if beats_ensemble:
        confirm = _h2h(
            "confirm: best elite vs ensemble", ref, DMC_ENSEMBLE, PPO_CLASSES
        )
        runs.append(confirm)
        beats_ensemble = confirm["ci_lo"] > 0
    passed = beats_ensemble and vs_distilled["ci_lo"] > 0

    if gen_cycle == check_every:
        # One-off: do EMA snapshots suit a policy as they do a Q-net?
        runs.append(
            _h2h(
                "best elite EMA vs live weights",
                ref,
                _live_ref(snapshot_dir, gen_cycle),
                PPO_CLASSES,
                PPO_CLASSES,
                EMA_CHECK_GAMES,
            )
        )

    previous = [
        record
        for cycle in range(check_every, gen_cycle, check_every)
        if (record := _load_gate2(snapshot_dir, cycle))
    ]
    ever_passed = passed or any(r["passed"] for r in previous)
    worse = vs_distilled["ci_hi"] < 0
    regression = worse and previous and previous[-1]["worse_than_distilled"]
    no_progress = (
        not ever_passed and gen_cycle >= GIVE_UP_CYCLE and vs_distilled["ci_lo"] <= 0
    )
    give_up = bool(regression or no_progress)

    verdict = "PASS" if passed else "not passed"
    if give_up:
        reason = "regression" if regression else "no progress"
        verdict += f" | GIVE UP ({reason})"
    lines = [
        f"\n### GATE 2, CYCLE {gen_cycle} ###",
        *(_line(s) for s in runs),
        f"  -> {verdict}",
    ]
    _log(
        snapshot_dir,
        f"gate2_{gen_cycle}",
        {
            "cycle": gen_cycle,
            "passed": passed,
            "worse_than_distilled": worse,
            "give_up": give_up,
            "runs": runs,
        },
        lines,
    )
    return give_up


def main():
    # python -m ppo_loop.gates_ppo gate1 | gate2 <cycle>
    args = sys.argv[1:]
    if args and args[0] == "gate2":
        check_gate2(PPO_DIR, int(args[1]))
    else:
        run_gate1()


if __name__ == "__main__":
    main()
