English | [中文](../zh-TW/DATA_DICTIONARY.md)

# Data dictionary

All files in `results/la2/`, with the step that writes them (see [EXPERIMENTS.md](EXPERIMENTS.md)). Files marked **(archive)** are not in the Git repository; they are in the large-data archive `learnable-ants-state-tokens-large-data.zip`, attached to the v1.0.0 GitHub release. Unzipping it in the repository root puts them in place.

Pickle files (`.pkl`) are written with Python's `pickle` module. Load only files from this repository or its archive.

## Token sequences

A *trip* (episode) is a list of token ids (`miat_ants.la2.vocab.STOI` / `ITOS`). It starts with the trip-start token `.` (id 0) and continues in the dual format, `o s a o s a …`: the perception, the teacher's control state, and the action at each step. A trip ends when the ant drops food at the nest (`D`) or after 400 steps (time-out). The bo and aux views are obtained with `miat_ants.la2.pretrain.as_mode(trips, "bo")`, which drops the state tokens.

| Group | Tokens |
|---|---|
| Actions (10) | `F L R` move forward / forward-left / forward-right; `f l r` the same while laying pheromone; `U` U-turn; `K` scan in place; `T` take food; `D` drop food at the nest |
| Perception, searching (10) | `X` rich food underfoot; `Y` poor food underfoot; `A B C` food ahead / ahead-left / ahead-right; `P Q S` pheromone ahead / ahead-left / ahead-right; `N` no cue; `W` wall ahead |
| Perception, food-site vector (5) | `g h j k` remembered food site ahead / right / behind / left; `z` at the remembered site, no food |
| Perception, carrying (10) | `0` in the nest; `w` wall ahead; `1 2 3 4` nest ahead / right / behind / left on a weak trail; `5 6 7 8` the same on a strong trail |
| Control states (8) | `ε` exploration; `φ` trail following; `ν` vector navigation; `σ` systematic search; `κ` scanning; `υ` U-turn; `η` carrying home; `δ` dropping food |

## Teacher corpora

| File | Step | Content |
|---|---|---|
| `corpus.pkl` | pretrain | 10,783 teacher trips (two-patch and double bridge, simulation seeds 500–505). `miat_ants.la2.pretrain.split(trips, seed=0)` gives the 9,705 training and 1,078 held-out trips used throughout |
| `corpus_f05.pkl`, `corpus_f10.pkl`, `corpus_f20.pkl` **(archive)** | noise | the same collection with perception tokens flipped at rate 0.05, 0.1 or 0.2 |
| `r1/gen_<env>_eps.pkl` | offline (`gen`) | teacher trips from each test environment (`barrier`, `food_fast`, `food_slow`, `decay_fast`, `decay_slow`) and from a fresh two-patch run (`train`) |
| `r2_stability_eps.pkl` | colonies | the 150 fixed two-patch teacher trips (simulation seed 900) of the state-stability test |

## Offline results

| File | Step | Content |
|---|---|---|
| `e0_pretrain.json` | pretrain | one entry per bo/dual model, seeds 0–4: held-out action NLL and accuracy, settings, parameters, training time |
| `e124.json` | entropy | `e1`: conditional entropies of the state (bits; Table S5) and state occupancy; `ngram`: $n$-gram accuracy for n = 1–16; `seeds`: per-seed free-running tracking, probes, intervention, and clustering purity |
| `e124_f05.json`, `e124_f10.json`, `e124_f20.json` | noise | the same for the noisy corpora (three seeds) |
| `e3_summary.csv` | ablation | one row per noise level and seed, from `e124*.json`: `flip`, `seed`, `dual_free` (with `_sigma`, `_phi`), `bo_probe`, `obs_probe`, `ngram8` (with `_sigma`, `_phi`), `ngram1`, `action_acc_bo`, `action_acc_dual`, `follow` |
| `e5.json`, `e5_jobs/*.json` | ablation | one entry per configuration, format and seed |
| `e5_summary.csv` | ablation | `config` (`ctx4`, `ctx8`, `ctx32`, `embd32`, `data10`, `data30`, `main`), `mode`, `seed`, `action_nll`, `action_acc`, and for dual models free-running tracking `free`, `free_sigma`, `free_phi` (Table S12) |
| `r1_main.json` | offline | `summary`: accuracy per method, overall (`all`), per state, and balanced over states; for learned methods also `min`, `max` and `per_seed`. `pairs`: trip-level paired bootstrap for `"<a>\|<b>\|<state or all>"` with mean difference, 95% interval and n. `eval`: action prediction per format. `intervene`: one entry per seed (`n`, `follow_frac`, `js_orig`, `js_swapped`). `occupancy_heldout`: state shares on the held-out trips |
| `r1_perseed.json` | offline | list over training seeds; per seed the overall and per-state accuracy of `dual`, `aux`, `gru_aux`, `bo_probe`, `gru_bo_probe`, `rand_probe` (Table S8) |
| `r1_ctx.json` | offline | `rows`: tracking of aux and dual models by context length; `pairs`: dual − aux per context (Table S11) |
| `r1_noise.json` | offline | `rows`: tracking per noise level and method; `pairs`: dual against $n$-gram and filter (Table S9) |
| `r1_gen.json` | offline | per environment: `summary`, `pairs`, `n_trips`, `occupancy` (Table S20) |
| `r1_act.json` | offline | action agreement and NLL per format on every held-out step; dual both with the teacher's states and with its own (Table S4) |
| `r1/seed_<eval>_<k>.pkl` **(archive)** | offline | per training seed k and evaluation set: per-trip state predictions of every learned method (`pred`), action evaluation (`eval`), intervention (`intervene`) |
| `r1/det_<eval>.pkl` **(archive)** | offline | per-trip predictions of the $n$-grams and the finite-state filter |
| `r1/ctx_<mode>_<ctx>_<k>.pkl`, `r1/noise_f<p>_<k>.pkl` **(archive)** | offline | per-trip predictions of the context and noise parts |

Method keys: `ngram1` (lookup) to `ngram16`, `filter`, `obs_probe`, `rand_probe`, `bo_probe`, `gru_bo_probe`, `aux`, `gru_aux`, `dual`. Accuracies are fractions (0–1).

## Colony results

| File | Step | Content |
|---|---|---|
| `r2_jobs/<cond>__<env>__<seed>.pkl` **(archive)** | colonies | one record per colony: the columns of `r2_per_seed.csv` plus `deliv_curve` (cumulative delivery every 50 steps) |
| `r2_per_seed.csv` | analysis | 700 rows, one per colony (columns below) |
| `r2_delivery_curves.csv` | analysis | cumulative food delivered every 50 steps: `cond`, `env`, `seed`, `step` (50–3,000), `delivered`; 42,000 rows (not plotted in the article; provided for re-analysis) |
| `r3_colony.json` | analysis | `tables`: per environment and condition, mean and 95% interval of each column; `paired`: yield differences between paired conditions (`env`, `a`, `b`, `mean`, `ci_lo`, `ci_hi`, `rel_pct`, `p`, `p_holm`, `n`); `stab_paired`: the same for stability, σ occupancy, practiced σ and time-outs |
| `r3_summary.txt` | analysis | the printed summary of `la2_r3_analysis.py` |
| `e7_per_seed.csv` | teacher_search | 20 two-patch colonies each of `teacherS`, `dual_learnA`, `teacher_nomem`, and frozen dual colonies 10–19; adds `sig_in_success`, `sig_in_fail`, `sigfrac_success`, `sigfrac_all`, `n_success_eps`, `n_fail_eps` (Table S19) |

Conditions (`cond`): `teacher`; `teacher_nomem` (teacher without food-site memory); `bo_frozen`, `aux_frozen`, `dual_frozen` (no learning); `bo_learn`, `aux_learn`, `dual_learn` (online self-imitation, trip-weighted); `dual_learn_step` (step-weighted sampling); `dual_learn_sigma` (search-balanced sampling); `dual_learn_reh` (a quarter of each batch from the teacher's training trips); `dual_learn_rehNS`, `dual_learn_rehS` (teacher rehearsal only from trips without or with σ); `dual_learn_A` (dual learning of actions only).

Environments (`env`): `two_patch`, `double_bridge`, `barrier`, `food_fast`, `food_slow`, `decay_fast`, `decay_slow`.

Columns of `r2_per_seed.csv`:

| Column | Meaning |
|---|---|
| `cond`, `env`, `seed` | condition, environment, simulation seed |
| `train_seed` | training seed of the starting weights (`seed mod 10`; −1 for the teacher) |
| `delivered` | food units delivered to the nest in 3,000 steps |
| `takes_rich`, `takes_poor`, `rich_share` | food taken from rich and poor patches, and the rich share |
| `gate_short`, `gate_long`, `short_share` | corridor crossings on the short and long bridge, and the short share (double bridge only) |
| `timeouts`, `success_eps` | abandoned and successful trips |
| `oa_legal`, `osa_legal`, `trans_legal` | share of perception→action, (perception, state)→action and state→state transitions that occur in the teacher corpus |
| `occ_<state>` | share of steps spoken (or, for the teacher, spent) in each state |
| `mark_q1`, `mark_q0` | marking rate on carrying moves along a trail, for rich (q1) and poor (q0) food |
| `mark_q<q>_weak`, `mark_q<q>_strong` | the same on weak and strong trails |
| `mark_q<q>_near`, `mark_q<q>_far` | within 8 steps of taking the food, or later |
| `mark_searched<0/1>_q<q>` | after a trip without or with a visit to an empty remembered site |
| `dispersion` | mean pairwise Jensen–Shannon divergence between the ants' action distributions on fixed probe contexts at the end of the run |
| `sig_in_success`, `sig_in_all` | share of successful trips, and of all trips, that contain σ |
| `stab_all`, `stab_<state>` | state stability: free-running tracking with the ants' final weights on the 150 fixed teacher trips, overall and per state (two-patch, aux and dual only) |
| `practiced_sigma`, `practiced_sigma_late` | share of σ among the state targets practiced in online learning, over the whole run and over its second half (dual and aux learning only) |
| `seconds` | run time |

Empty cells mean that a measure does not apply (for example `occ_*` for bo ants, or `short_share` on the two-patch map).

## Weights

`miat_ants/la2/weights/` holds every trained model as `<name>.npz` (parameters) and `<name>.json` (settings, held-out action NLL and accuracy, state accuracy, parameters, training time, and the training curve). Names are listed in [`miat_ants/la2/README.md`](../../miat_ants/la2/README.md). Load transformers with `miat_ants.microgpt.MicroGPT.load(path)` and GRUs with `miat_ants.la2.gru.GRUModel.load(path)`. `miat_ants/foraging/weights/` holds the 24-token baseline models of the companion study.
