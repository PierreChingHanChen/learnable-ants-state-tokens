"""Run the experiments of "Self-Generated State Tokens as Memory in Learnable Foraging Agents" in order.

Usage
  python run_all.py                    # every step (several hours on two CPU cores)
  python run_all.py figures tables     # only regenerate figures and supplementary tables from stored results
  python run_all.py --list             # list the steps

Each step skips work whose output already exists (trained weights, cached job files), so an interrupted
run can be resumed by running the same command again.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(ROOT, "experiments")

# (name, [(script, environment), ...], what it produces)
NOISE = [("la2_e124_offline.py", {"LA2_CORPUS": os.path.join(ROOT, "results", "la2", f"corpus_{t}.pkl"),
                                  "LA2_WTAG": f"_{t}", "LA2_TAG": f"_{t}", "LA2_SEEDS": "3"}) for t in ("f05", "f10", "f20")]
STEPS = [
    ("pretrain", [("la2_e0_pretrain.py", {})], "teacher corpus; bo and dual models, training seeds 0-4"),
    ("train", [("la2_r0_train.py", {})],
     "bo/dual seeds 5-9; aux seeds 0-9; context ablation; GRU-bo and GRU-aux seeds 0-9"),
    ("noise", [("la2_e3_noise.py", {})] + NOISE, "noisy corpora and models; their first offline evaluation (e124_f*.json)"),
    ("entropy", [("la2_e124_offline.py", {})], "state entropies and clustering purity (Table S5, Section 4.6)"),
    ("ablation", [("la2_e5_ablation.py", {}), ("la2_e35_summary.py", {})],
     "further ablations (Table S12) and the e3/e5 summary CSVs"),
    ("offline", [("la2_r1_offline.py", {"LA2_R1_PART": "main,ctx,noise,gen,act"})],
     "paired offline tracking, context, noise, generalization, action prediction (Figs. 2, 3, 5A; Tables S4, S6-S11, S20)"),
    ("colonies", [("la2_r2_colony.py", {"LA2_R2_PLAN": "main;reh_split;bridge;gen;gen_learn"})],
     "colony experiments (Figs. 4, 5B-C; Tables S13-S18, S21)"),
    ("teacher_search", [("la2_e7_followup.py", {})], "search share in teacher trips (Table S19)"),
    ("analysis", [("la2_r3_analysis.py", {})], "colony statistics -> results/la2/r3_colony.json, per-seed CSV, curves"),
    ("figures", [("la2_r4_figures.py", {})], "Figures 2-5 -> figures/paper/"),
    ("tables", [("la2_r5_supp_tables.py", {})], "supplementary tables (LaTeX) -> results/la2/supp_tables/"),
]


def main(argv):
    if "--list" in argv:
        for name, runs, what in STEPS:
            print(f"{name:15s} {', '.join(dict.fromkeys(r[0] for r in runs)):40s} {what}")
        return
    wanted = [a for a in argv if not a.startswith("-")] or [s[0] for s in STEPS]
    unknown = set(wanted) - {s[0] for s in STEPS}
    if unknown:
        sys.exit(f"unknown step(s): {', '.join(sorted(unknown))}; use --list")
    for name, runs, what in STEPS:
        if name not in wanted:
            continue
        print(f"\n=== {name}: {what}", flush=True)
        for script, env in runs:
            args = [sys.executable, script] + (["fig2", "fig3", "fig4", "fig5"] if name == "figures" else [])
            subprocess.run(args, cwd=EXP, env={**os.environ, **env}, check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
