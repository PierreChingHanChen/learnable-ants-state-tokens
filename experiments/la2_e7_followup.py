"""LA2 E7：教師回合中系統搜尋的比例（補充表 S19；two_patch，20 個種子），以及早期單一模型設計下的兩項追加條件
(a) teacherS：重跑教師，記錄搜尋狀態 σ 在成功／逾時回合中的比例（自我模仿的選擇偏差）
(b) dual_learnA：dual 線上學習只學行動、不學狀態詞
(c) teacher_nomem：關閉食物地點記憶，檢查品質調節能否讓群體選中優質食物（Beckers 1993）"""
import os, pickle
import pandas as pd
from common import pmap
import la2_e6_colony as E

SEEDS = int(os.environ.get("LA2_E7_SEEDS", "20"))

if __name__ == "__main__":
    os.makedirs(E.JOBDIR, exist_ok=True)
    jobs = [(c, "two_patch", s) for c in ("dual_learnA", "teacherS", "teacher_nomem") for s in range(SEEDS)]
    jobs += [("dual_frozen", "two_patch", s) for s in range(10, SEEDS)]     # 凍結對照補到 20 個種子
    res = pmap(E.job, jobs)
    pd.DataFrame([{k: v for k, v in r.items() if k != "deliv_curve"} for r in res]).to_csv(
        os.path.join(E.OUT, "e7_per_seed.csv"), index=False)
    print("done", len(res))
