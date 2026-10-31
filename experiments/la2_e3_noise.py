"""LA2 E3：感知翻轉雜訊 p ∈ {0.05, 0.1, 0.2}。每級重新收集教師語料，訓練 bo／dual（各 3 個種子）。
離線評估由 la2_e124_offline.py 以 LA2_CORPUS／LA2_WTAG／LA2_TAG 環境變數執行。"""
import os, pickle
from common import pmap, RESULTS
from miat_ants.la2 import pretrain as P
from miat_ants.la2.system import base_path

OUT = os.path.join(RESULTS, "la2")
LEVELS = [float(x) for x in os.environ.get("LA2_FLIPS", "0.05,0.1,0.2").split(",")]
SEEDS = int(os.environ.get("LA2_SEEDS", "3"))


def tag(p):
    return f"_f{int(round(p * 100)):02d}"


def collect(p):
    path = os.path.join(OUT, f"corpus{tag(p)}.pkl")
    if not os.path.exists(path):
        pickle.dump(P.collect(obs_flip=p), open(path, "wb"))
    return path


def job(args):
    p, mode, seed = args
    if os.path.exists(base_path(mode, f"{tag(p)}_s{seed}")):
        return {"skipped": (p, mode, seed)}
    eps = pickle.load(open(os.path.join(OUT, f"corpus{tag(p)}.pkl"), "rb"))
    tr, va = P.split(eps, seed=0)
    _, st = P.train(mode, tr, va, seed=seed, out=base_path(mode, f"{tag(p)}_s{seed}"))
    return {k: v for k, v in st.items() if k != "curve"}


if __name__ == "__main__":
    pmap(collect, LEVELS)
    for r in pmap(job, [(p, m, s) for p in LEVELS for m in ("bo", "dual") for s in range(SEEDS)]):
        print(r)
