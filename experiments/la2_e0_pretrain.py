"""LA2 E0：收集教師語料，訓練 bo／dual 主模型（各 5 個種子），ctx_steps = 16。"""
import os, pickle, json
import numpy as np
from common import pmap, RESULTS
from miat_ants.la2 import pretrain as P
from miat_ants.la2.system import base_path

OUT = os.path.join(RESULTS, "la2")
os.makedirs(OUT, exist_ok=True)
CORPUS = os.path.join(OUT, "corpus.pkl")
SEEDS = int(os.environ.get("LA2_SEEDS", "5"))


def job(args):
    mode, seed = args
    eps = pickle.load(open(CORPUS, "rb"))
    tr, va = P.split(eps, seed=0)
    _, st = P.train(mode, tr, va, seed=seed, out=base_path(mode, f"_s{seed}"))
    return st


def main():
    if not os.path.exists(CORPUS):
        eps = P.collect()
        pickle.dump(eps, open(CORPUS, "wb"))
    res = pmap(job, [(m, s) for m in ("bo", "dual") for s in range(SEEDS)])
    for st in res:
        print({k: v for k, v in st.items() if k != "curve"})
    for m in ("bo", "dual"):                        # 種子 0 作為預設基礎模型
        import shutil
        shutil.copy(base_path(m, "_s0"), base_path(m))
    json.dump([{k: v for k, v in st.items() if k != "curve"} for st in res],
              open(os.path.join(OUT, "e0_pretrain.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
