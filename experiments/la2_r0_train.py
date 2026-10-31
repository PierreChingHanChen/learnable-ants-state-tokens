"""R0：訓練種子 5–9 與比較模型（種子 0–4 由 la2_e0_pretrain.py 訓練）。
  * bo／dual：種子 5–9（與種子 0–4 相同流程）→ 共 10 個訓練種子
  * aux（監督狀態、不回饋）：種子 0–9
  * 上下文長度消融：aux 與 dual 的 4、8 步上下文，種子 0–2（權重保存，以便在全部保留回合上配對評估）
  * GRU-bo、GRU-aux（約 4,979 參數，完整回合 BPTT）：種子 0–9"""
import os, pickle, json
import numpy as np
from common import pmap, RESULTS
from miat_ants.la2 import pretrain as P, gru as G
from miat_ants.la2.system import base_path, WEIGHTS_DIR

OUT = os.path.join(RESULTS, "la2")
CORPUS = os.path.join(OUT, "corpus.pkl")
N_SEEDS = int(os.environ.get("LA2_NSEEDS", "10"))


def gru_path(obj, seed):
    return os.path.join(WEIGHTS_DIR, f"gru_{obj}_s{seed}.npz")


def job(args):
    kind, mode, ctx, seed = args
    out = gru_path(mode, seed) if kind == "gru" else base_path(mode, (f"_ctx{ctx}" if ctx != 16 else "") + f"_s{seed}")
    if os.path.exists(out):
        return {"skipped": out}
    eps = pickle.load(open(CORPUS, "rb"))
    tr, va = P.split(eps, seed=0)
    if kind == "gru":
        _, st = G.train(mode, tr, va, seed=seed, out=out)
    else:
        _, st = P.train(mode, tr, va, ctx_steps=ctx, seed=seed, out=out)
    return {k: v for k, v in st.items() if k != "curve"}


if __name__ == "__main__":
    jobs = [("tf", "dual", 16, s) for s in range(5, N_SEEDS)]
    jobs += [("tf", "dual", c, s) for c in (4, 8) for s in range(3)]
    jobs += [("tf", "aux", 16, s) for s in range(N_SEEDS)]
    jobs += [("tf", "aux", c, s) for c in (4, 8) for s in range(3)]
    jobs += [("tf", "bo", 16, s) for s in range(5, N_SEEDS)]
    jobs += [("gru", o, 0, s) for o in ("aux", "bo") for s in range(N_SEEDS)]
    for r in pmap(job, jobs, workers=2):
        print(r, flush=True)
