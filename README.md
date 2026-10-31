English | [中文](README.zh-TW.md)

<!-- After the first Zenodo release, add the DOI badge here, e.g.
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

# Learnable ants: Self-generated state tokens as memory

Code, data, and figures for the study **"Self-Generated State Tokens as Memory in Learnable Foraging Agents"** (Ching-Han Chen, National Central University, Taiwan).

Small sequence models can drive simulated foragers, but what such agents represent about their own behavioral state is rarely testable. This repository builds a 44-token language for ant foraging whose perception, action, and control-state tokens are drawn from published ant behavior (quality- and crowding-dependent trail laying, food-site vectors, systematic search, scanning, U-turns), a rule-based *ethological teacher* that speaks it, and transformer ants of about 5,000 parameters that either only act (**bo**), predict their state without reading it back (**aux**), or first state their control state and then act (**dual**). It compares the self-spoken state token with a parameter-matched GRU and a finite-state Bayesian filter, and studies colonies of 30 such ants that keep learning from their own successful trips.

| Question | Finding |
|---|---|
| Does speaking the state recover a hidden state? | Free-running dual models tracked the teacher's hidden state at 98.7% of steps on 1,078 held-out trips, against 97.1% for an eight-step history lookup; systematic search at 100% against 88.7%. |
| Supervision or feedback? | Predicting the state built most of the representation (aux 98.0%); reading it back carried it beyond the context window (four-step context: search tracked at 100% with feedback, 78% without). |
| Is a token a better memory? | No. A GRU of the same size tracked states slightly better (99.6%) and a finite-state filter as well (98.8%), and more robustly under perceptual noise. The token makes a windowed model stateful with a legible state; replacing it redirected actions in 94% of cases. |
| What happens when colonies keep learning? | When food depleted and moved, online self-imitation lowered yield by 3–7% for all formats and the spoken states drifted. Practicing more of the ants' own searches did not stop the drift; rehearsing the teacher's searches did and recovered up to 60% of the lost food. |

The article has been submitted; its reference will be added here on publication.

## Contents

```
miat_ants/              simulation platform (Python + numpy)
  core/                 GRAFCET engine, event contract, shared pheromone medium, metrics
  microgpt/             numpy transformer (batched over agents, hand-written gradients, Adam)
  foraging/             foraging platform and 24-token baseline of Chen (2026)
  la2/                  this study: 44-token dual-alphabet vocabulary, world, teacher, cores
    vocab.py            vocabulary; bo / aux / dual formats; trip-, step- and search-weighted samplers
    world.py            World2 (food quality, food-site vectors, search, U-turns, scanning) and the test maps
    teacher.py          ethological teacher with eight behavioral states
    core.py             transformer ants (bo, aux, dual), online self-imitation, teacher rehearsal
    gru.py              parameter-matched GRU (full backpropagation through time)
    metrics.py          n-grams, finite-state (IO-HMM) filter, free-running tracking, probes, interventions, legality
    weights/            all trained models (transformers, GRUs, context and noise variants) with training metadata
experiments/            one script per step (see "Reproducing the article")
results/la2/            teacher corpus, summary results and per-seed data used in the article
figures/paper/          Figures 1–5 of the article (PDF + PNG)
docs/en/, docs/zh-TW/   experiment guide and data dictionary in English and Traditional Chinese
tests/                  14 automated tests
run_all.py              runs every step in order
```

## Installation

Python 3.10 or newer (tested with Python 3.13).

```bash
pip install -r requirements.txt
python -m pytest -q          # 14 tests, about 10 s
```

## Quick start

Regenerate all figures of the article and all supplementary tables from the stored results (about one minute):

```bash
python run_all.py figures tables     # -> figures/paper/ and results/la2/supp_tables/
```

Run one ant colony (30 dual ants with online learning on the two-patch map):

```python
from miat_ants.la2.system import LA2Config, LA2System, base_path

r = LA2System(LA2Config(map_name="two_patch", core="gpt", mode="dual", learn=True, seed=0,
                        base_weights=base_path("dual", "_s0"), max_steps=3000)).run()
print(r.delivered)
```

Track the teacher's hidden state with a trained dual model (free-running: every state token in the context is the model's own prediction):

```python
import pickle
from miat_ants.la2 import metrics as M, pretrain as P
from miat_ants.la2.system import base_path
from miat_ants.microgpt import MicroGPT

train, held_out = P.split(pickle.load(open("results/la2/corpus.pkl", "rb")), seed=0)
dual = MicroGPT.load(base_path("dual", "_s0"))
pred = M.free_run_multi(dual, held_out[:100], [0])[0]
```

## Reproducing the article

`python run_all.py` runs every step in order (several hours on two CPU cores); `python run_all.py --list` lists the steps, and any subset can be run by name. Steps skip work whose output already exists, so an interrupted run resumes where it stopped.

| Step | Script | Produces |
|---|---|---|
| `pretrain` | `la2_e0_pretrain.py` | teacher corpus; bo and dual models, training seeds 0–4 |
| `train` | `la2_r0_train.py` | bo and dual seeds 5–9; aux seeds 0–9; context ablation (4, 8 steps); GRU-bo and GRU-aux seeds 0–9 |
| `noise` | `la2_e3_noise.py`, `la2_e124_offline.py` | noisy corpora and models, first offline evaluation (paired evaluation for Fig. 3C and Table S9 is in `offline`) |
| `entropy` | `la2_e124_offline.py` | state entropies (Table S5) and clustering purity (Section 4.6) |
| `ablation` | `la2_e5_ablation.py`, `la2_e35_summary.py` | further ablations (Table S12); summary CSVs |
| `offline` | `la2_r1_offline.py` | paired offline tracking, context length, noise, generalization, action prediction (Figs. 2, 3, 5A; Tables S4, S6–S11, S20) |
| `colonies` | `la2_r2_colony.py` | colony experiments on both maps, rehearsal schemes, state stability, test environments (Figs. 4, 5B–C; Tables S13–S18, S21) |
| `teacher_search` | `la2_e7_followup.py` | share of search in the teacher's trips (Table S19) |
| `analysis` | `la2_r3_analysis.py` | colony statistics with Holm correction → `results/la2/r3_colony.json`; per-colony CSV and delivery curves |
| `figures` | `la2_r4_figures.py` | Figures 2–5 → `figures/paper/` (Figure 1 is a diagram) |
| `tables` | `la2_r5_supp_tables.py` | Supplementary Tables S4, S6–S11, S13–S21 (LaTeX) → `results/la2/supp_tables/` |

See [docs/en/EXPERIMENTS.md](docs/en/EXPERIMENTS.md) for settings and run times, and [docs/en/DATA_DICTIONARY.md](docs/en/DATA_DICTIONARY.md) for every file in `results/la2/`.

## Data

This repository contains everything needed to regenerate the figures and tables: the teacher corpus (`results/la2/corpus.pkl`, 10,783 trips), all trained weights, summary results (`r1_*.json`, `r3_colony.json`), per-colony results (`r2_per_seed.csv`, 700 colonies), and delivery curves (`r2_delivery_curves.csv`).

Intermediate files that can be regenerated by the scripts are not tracked in Git because they are large (about 160 MB unpacked): the per-trip predictions of every method and seed (`results/la2/r1/*.pkl`), the raw colony records (`results/la2/r2_jobs/*.pkl`), and the noisy teacher corpora (`results/la2/corpus_f*.pkl`). They are attached to the GitHub release as `learnable-ants-state-tokens-large-data.zip` (5 MB) and archived with it at Zenodo (see Citation). Unzipping that file in the repository root restores them; the scripts then use them instead of recomputing.

## License

- Code (`miat_ants/` except `weights/`, `experiments/`, `tests/`, `run_all.py`): [MIT License](LICENSE).
- Data, figures, model weights, and documentation (`results/`, `figures/`, `miat_ants/la2/weights/`, `miat_ants/foraging/weights/`, `docs/`, README files): [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE-DATA).

## Citation

If you use this code or data, please cite the archived release (the DOI will be added after the first Zenodo release):

> Chen, C.-H. (2026). *Learnable ants: Self-generated state tokens as memory (code and data)* (v1.0.0). Zenodo. https://doi.org/[to be added]

Machine-readable metadata are in [CITATION.cff](CITATION.cff). The companion study on colony-level coordination is at [learnable-ants-cwc](https://github.com/PierreChingHanChen/learnable-ants-cwc).

## Contact

Questions and bug reports: please open an issue.
