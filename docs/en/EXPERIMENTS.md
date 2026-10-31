English | [中文](../zh-TW/EXPERIMENTS.md)

# Experiment guide

This guide lists, for every step of `run_all.py`, what the step does, its settings, and where its output goes. Section numbers refer to the article and "Table S…" to its Supplementary Material. All scripts are in `experiments/` and are run from that directory (`run_all.py` does this for you); all outputs go to `results/la2/` unless stated otherwise.

Every step skips work whose output already exists: trained weights in `miat_ants/la2/weights/`, per-trip predictions in `results/la2/r1/`, colony records in `results/la2/r2_jobs/`, ablation jobs in `results/la2/e5_jobs/`. To recompute something, delete its output first. Parallel steps use two worker processes by default (`LA2_WORKERS` for the colony step).

## Common settings

| Setting | Value |
|---|---|
| Vocabulary | 44 tokens: 25 perception, 10 action, 8 control-state tokens and the trip-start token (`miat_ants/la2/vocab.py`) |
| Formats | **bo** `. o a o a …`; **aux** the bo stream with an additional state head at perception positions (not fed back); **dual** `. o s a o s a …` |
| Teacher corpus | teacher colonies on the two-patch and double-bridge maps, simulation seeds 500–505, 1,500 steps each: 10,783 trips, split 9,705 training / 1,078 held-out trips (`pretrain.split`, seed 0) |
| Transformers | microGPT in numpy: embedding width 16, context 16 steps, about 5,000 parameters (bo and aux 4,992; dual 5,248) |
| Transformer training | 3,000 Adam steps, batch 64, learning rate 10⁻² decaying linearly |
| GRU | hidden size 25, 4,979 parameters; full backpropagation through time on whole trips; 3,000 Adam steps of 64 trips, learning rate 10⁻² decaying linearly, gradient clipping at 1 |
| Training seeds | 10 per format (0–9); 3 per cell for the context-length ablation and the noise runs |
| Colonies | 30 ants, shared pheromone medium (ρ = 0.97), 3,000 steps; simulation seed k starts from training seed k mod 10 |
| Online learning | each ant keeps its 20 most recent successful trips; every 50 steps four Adam steps (learning rate 3·10⁻³, batch 16) |
| Statistics | 95% percentile bootstrap (10,000 resamples); trip-level paired bootstrap offline; Wilcoxon signed-rank tests with Holm correction for colony yield |

Run times below were measured on two CPU cores (Python 3.13, numpy 2.5).

## Steps

### `pretrain` — teacher corpus and first models (`la2_e0_pretrain.py`)
Collects the teacher corpus (`corpus.pkl`) if it does not exist and trains bo and dual models for training seeds 0–4 (`LA2_SEEDS`, default 5). Weights: `miat_ants/la2/weights/la2_{bo,dual}_s{0..4}`. Summary: `e0_pretrain.json`. About 2 min per model.

### `train` — remaining seeds and comparison models (`la2_r0_train.py`)
Trains bo and dual seeds 5–9, aux seeds 0–9, the context-length ablation (aux and dual with 4- and 8-step contexts, seeds 0–2), and GRU-bo and GRU-aux seeds 0–9 (`LA2_NSEEDS`, default 10). About 52 models of 2–3 min each, i.e. about 2 core-hours.

### `noise` — perceptual noise (`la2_e3_noise.py`, then `la2_e124_offline.py`)
For each flip probability p ∈ {0.05, 0.1, 0.2} (`LA2_FLIPS`), re-collects the teacher corpus with perception tokens replaced at rate p by a random token of the same group (`corpus_f05.pkl`, …; the teacher acts on the noisy tokens), and trains bo and dual models, seeds 0–2 (`la2_{bo,dual}_f{05,10,20}_s{k}`). It then runs a first offline evaluation of these models (`e124_f{05,10,20}.json`). The paired evaluation used in the article (Fig. 3C, Table S9) is part of the `offline` step.

### `entropy` — state entropies and clustering (`la2_e124_offline.py`)
Conditional entropies of the teacher's state (Section 4.1, Table S5: `e124.json`, key `e1`), $n$-gram baselines, and clustering purity of bo activations (Section 4.6), for training seeds 0–4.

### `ablation` — further ablations (`la2_e5_ablation.py`, then `la2_e35_summary.py`)
Context 4, 8 and 32 steps, embedding width 32, and 10% or 30% of the training trips, for bo and dual, seeds 0–2 (Table S12). Jobs are cached in `e5_jobs/`; results in `e5.json` and `e5_summary.csv`. `la2_e35_summary.py` also writes `e3_summary.csv` from `e124*.json` and prints both summaries as Markdown.

### `offline` — paired offline evaluation (`la2_r1_offline.py`)
Every method is evaluated on the same 1,078 held-out trips (54,523 steps) and compared trip by trip. `LA2_R1_PART` selects the parts:

| Part | Content | Output | Article |
|---|---|---|---|
| `main` | lookup and $n$-grams (n = 1, 2, 4, 8, 16), finite-state filter, probes (bo, GRU-bo, random-init transformer, current perception), aux, GRU-aux, dual free-running; intervention test (4,000 cases per seed) | `r1_main.json`, `r1_perseed.json`, `r1/seed_heldout_*.pkl`, `r1/det_heldout.pkl` | Fig. 2; Tables S6–S8, S10 |
| `ctx` | aux and dual with 4-, 8- and 16-step contexts, seeds 0–2 | `r1_ctx.json`, `r1/ctx_*.pkl` | Fig. 3A–B; Table S11 |
| `noise` | dual, $n$-gram and filter on the noisy corpora | `r1_noise.json`, `r1/noise_*.pkl` | Fig. 3C; Table S9 |
| `gen` | all methods on teacher trips from the five test environments and from a fresh two-patch run | `r1_gen.json`, `r1/gen_*_eps.pkl`, `r1/seed_<env>_*.pkl` | Fig. 5A; Table S20 |
| `act` | top-1 action agreement and negative log-likelihood on every held-out step | `r1_act.json` | Section 4.1; Table S4 |

The `r1/*.pkl` files hold the per-trip predictions (about 150 MB in all). Only the `gen_*_eps.pkl` trip sets are kept in this repository; the rest are in the large-data archive.

### `colonies` — colony experiments (`la2_r2_colony.py`)
Closed-loop colonies of 30 ants for 3,000 steps. `LA2_R2_PLAN` lists the job plans, separated by semicolons:

| Plan | Conditions | Environment | Colonies |
|---|---|---|---|
| `main` | teacher, teacher_nomem, {bo, aux, dual}_{frozen, learn}, dual_learn_step, dual_learn_sigma, dual_learn_reh, dual_learn_A | two-patch | 12 × 20 |
| `reh_split` | dual_learn_rehNS (teacher rehearsal without σ trips), dual_learn_rehS (only σ trips) | two-patch | 2 × 20 |
| `bridge` | teacher, {bo, aux, dual}_{frozen, learn} | double bridge | 7 × 20 |
| `gen` | teacher, {bo, aux, dual}_frozen | barrier, food fast, food slow, decay fast, decay slow | 4 × 5 × 10 |
| `gen_learn` | {bo, aux, dual}_learn, dual_learn_reh | food fast, food slow | 4 × 2 × 10 |

That is 700 colonies in all, about 10 core-hours (a frozen colony takes 20–30 s, a learning dual colony about 100 s, a teacher colony about 10 s). In two-patch runs of aux and dual ants, the state-stability test runs the free-running tracking test with each ant's final weights on 150 fixed teacher trips (`r2_stability_eps.pkl`, simulation seed 900). One record per colony is written to `r2_jobs/<cond>__<env>__<seed>.pkl`.

Test environments: *barrier* is an unseen 36 × 36 map with two internal walls and three patches; *food fast* and *food slow* are the two-patch map with 10 or 90 food units per cell instead of 30; *decay fast* and *decay slow* use pheromone retention ρ = 0.94 or 0.99 instead of 0.97.

### `teacher_search` — search in the teacher's trips (`la2_e7_followup.py`)
Re-runs 20 teacher colonies on the two-patch map and records how often systematic search (σ) occurs in successful and failed trips, together with two earlier conditions (dual_learnA, teacher_nomem) and ten more frozen dual colonies (Table S19). Uses the colony function in `la2_e6_colony.py`. Output: `e7_per_seed.csv`.

### `analysis` — colony statistics (`la2_r3_analysis.py`)
Pairs conditions by simulation seed, computes bootstrap intervals and Wilcoxon tests, and applies Holm correction within each environment. Output: `r3_colony.json`; the printed summary is stored as `r3_summary.txt`. When the raw colony records `r2_jobs/*.pkl` are present it also rewrites `r2_per_seed.csv` and `r2_delivery_curves.csv`; otherwise it reads `r2_per_seed.csv`, so the statistics can be recomputed from this repository alone.

### `figures` — Figures 2–5 (`la2_r4_figures.py`)
Reads `r1_main.json`, `r1_ctx.json`, `r1_noise.json`, `r1_gen.json` and `r2_per_seed.csv` and writes `figures/paper/fig{2..5}_*.{pdf,png}` (set `FIGDIR` to write elsewhere). Figure 1 is a diagram and is provided as is.

### `tables` — supplementary tables (`la2_r5_supp_tables.py`)
Writes the LaTeX tables S4, S6–S11 and S13–S21 to `results/la2/supp_tables/` (set `TABDIR` to write elsewhere). The tables are identical to those in the Supplementary Material. Tables S1–S3 (vocabulary, teacher and model settings), S5 (entropies, from `e124.json`) and S12 (ablations, from `e5_summary.csv`) are typeset by hand.

## Tests

`python -m pytest -q` runs 14 tests (`tests/test_la2.py`, about 10 s): the vocabulary and the three formats, valid teacher sequences, food-site memory and giving up, U-turn and scanning actions, food-site vectors, a short dual colony, $n$-grams and entropies, the alignment of aux targets, search-balanced sampling, the cached forward pass, the GRU gradient against finite differences, the finite-state filter, an aux colony that records but does not feed back its states, and food respawning on the barrier map.
