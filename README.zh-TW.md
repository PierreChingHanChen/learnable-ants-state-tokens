[English](README.md) | 中文

<!-- 第一次在 Zenodo 發布後，在此加上 DOI 徽章，例如
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

# 可學習螞蟻：自我生成的狀態詞元作為記憶

本儲存庫收錄〈**Self-Generated State Tokens as Memory in Learnable Foraging Agents**〉（陳慶瀚，國立中央大學）研究的程式碼、資料與圖表。

小型序列模型可以驅動模擬覓食者，但這類代理人對自身行為狀態表徵了什麼，很少能被檢驗。本研究建立了一套 44 個詞元的螞蟻覓食語言，其感知、行動與控制狀態詞元都取自已發表的螞蟻行為研究（依食物品質與擁擠程度調節的路徑標記、食物地點向量、系統搜尋、掃視、U 形迴轉），並以規則式的「行為學教師」產生這種語言，訓練約 5,000 參數的 transformer 螞蟻。螞蟻分三種格式：只行動（**bo**）、預測狀態但不把它讀回（**aux**）、先說出控制狀態再行動（**dual**）。研究把自我說出的狀態詞元與參數量相同的 GRU 及有限狀態貝氏濾波器比較，並觀察 30 隻螞蟻組成、持續從自己成功回合中學習的蟻群。

| 問題 | 結果 |
|---|---|
| 說出狀態能否恢復隱藏狀態？ | 在 1,078 個保留回合上，自由追蹤的 dual 模型恢復教師隱藏狀態的比例為 98.7%，八步歷史查表為 97.1%；系統搜尋為 100% 對 88.7%。 |
| 是監督還是回饋？ | 預測狀態建立了大部分表徵（aux 98.0%）；把狀態讀回則把它帶到上下文窗口之外（四步上下文時，有回饋搜尋追蹤 100%，沒有回饋 78%）。 |
| 詞元是更好的記憶嗎？ | 不是。同樣大小的 GRU 追蹤得稍好（99.6%），有限狀態濾波器相當（98.8%），且在感知雜訊下更穩健。詞元的價值在於讓只有窗口的模型擁有可讀的狀態；替換它，94% 的情況下行動會跟著改變。 |
| 蟻群持續學習會怎樣？ | 食物會耗盡並移動時，線上自我模仿讓三種格式的產量都下降 3–7%，說出的狀態也發生漂移。多練習螞蟻自己的搜尋無法阻止漂移；複習教師的搜尋則可以，並挽回最多 60% 的損失。 |

論文已投稿，正式發表後會在此補上期刊引用資訊。

## 內容

```
miat_ants/              模擬平台（Python + numpy）
  core/                 GRAFCET 引擎、事件契約、共享費洛蒙介質、指標
  microgpt/             numpy 版 transformer（多代理人批次、手寫梯度、Adam）
  foraging/             覓食平台與 Chen (2026) 的 24 詞元基準
  la2/                  本研究：44 詞元雙字母語彙、世界、教師、核心
    vocab.py            語彙；bo／aux／dual 格式；依回合、依步數、搜尋平衡的取樣器
    world.py            World2（食物品質、食物地點向量、搜尋、U 形迴轉、掃視）與測試地圖
    teacher.py          具八個行為狀態的行為學教師
    core.py             transformer 螞蟻（bo、aux、dual）、線上自我模仿、教師複習
    gru.py              參數量相同的 GRU（完整時間反向傳播）
    metrics.py          n-gram、有限狀態（IO-HMM）濾波器、自由追蹤、探針、介入、合法率
    weights/            所有訓練好的模型（transformer、GRU、上下文與雜訊變體）及訓練紀錄
experiments/            每個步驟一支腳本（見「重現論文」）
results/la2/            教師語料、彙整結果與論文使用的逐種子資料
figures/paper/          論文圖 1–5（PDF + PNG）
docs/en/, docs/zh-TW/   實驗指南與資料字典（英文與繁體中文）
tests/                  14 個自動化測試
run_all.py              依序執行所有步驟
```

## 安裝

需要 Python 3.10 以上（以 Python 3.13 測試）。

```bash
pip install -r requirements.txt
python -m pytest -q          # 14 個測試，約 10 秒
```

## 快速開始

由已存的結果重新產生論文所有圖與補充表格（約一分鐘）：

```bash
python run_all.py figures tables     # -> figures/paper/ 與 results/la2/supp_tables/
```

執行一個蟻群（two-patch 地圖上 30 隻線上學習的 dual 螞蟻）：

```python
from miat_ants.la2.system import LA2Config, LA2System, base_path

r = LA2System(LA2Config(map_name="two_patch", core="gpt", mode="dual", learn=True, seed=0,
                        base_weights=base_path("dual", "_s0"), max_steps=3000)).run()
print(r.delivered)
```

以訓練好的 dual 模型追蹤教師的隱藏狀態（自由追蹤：上下文中的每個狀態詞都是模型自己的預測）：

```python
import pickle
from miat_ants.la2 import metrics as M, pretrain as P
from miat_ants.la2.system import base_path
from miat_ants.microgpt import MicroGPT

train, held_out = P.split(pickle.load(open("results/la2/corpus.pkl", "rb")), seed=0)
dual = MicroGPT.load(base_path("dual", "_s0"))
pred = M.free_run_multi(dual, held_out[:100], [0])[0]
```

## 重現論文

`python run_all.py` 依序執行所有步驟（兩核心 CPU 約數小時）；`python run_all.py --list` 列出步驟，也可以只指定部分步驟的名稱。已有輸出的工作會略過，因此中斷後重新執行同一指令即可接續。

| 步驟 | 腳本 | 產出 |
|---|---|---|
| `pretrain` | `la2_e0_pretrain.py` | 教師語料；bo 與 dual 模型，訓練種子 0–4 |
| `train` | `la2_r0_train.py` | bo 與 dual 種子 5–9；aux 種子 0–9；上下文消融（4、8 步）；GRU-bo 與 GRU-aux 種子 0–9 |
| `noise` | `la2_e3_noise.py`、`la2_e124_offline.py` | 雜訊語料與模型、第一次離線評估（圖 3C 與表 S9 的配對評估在 `offline`） |
| `entropy` | `la2_e124_offline.py` | 狀態熵（表 S5）與分群純度（4.6 節） |
| `ablation` | `la2_e5_ablation.py`、`la2_e35_summary.py` | 其他消融（表 S12）與摘要 CSV |
| `offline` | `la2_r1_offline.py` | 配對離線追蹤、上下文長度、雜訊、泛化、行動預測（圖 2、3、5A；表 S4、S6–S11、S20） |
| `colonies` | `la2_r2_colony.py` | 兩張地圖的群體實驗、複習方案、狀態穩定性、測試環境（圖 4、5B–C；表 S13–S18、S21） |
| `teacher_search` | `la2_e7_followup.py` | 教師回合中搜尋的比例（表 S19） |
| `analysis` | `la2_r3_analysis.py` | 群體統計（Holm 校正）→ `results/la2/r3_colony.json`；逐蟻群 CSV 與產量曲線 |
| `figures` | `la2_r4_figures.py` | 圖 2–5 → `figures/paper/`（圖 1 為示意圖） |
| `tables` | `la2_r5_supp_tables.py` | 補充表 S4、S6–S11、S13–S21（LaTeX）→ `results/la2/supp_tables/` |

各步驟的設定與執行時間見 [docs/zh-TW/EXPERIMENTS.md](docs/zh-TW/EXPERIMENTS.md)，`results/la2/` 中每個檔案的說明見 [docs/zh-TW/DATA_DICTIONARY.md](docs/zh-TW/DATA_DICTIONARY.md)。

## 資料

本儲存庫已包含重新產生所有圖表所需的資料：教師語料（`results/la2/corpus.pkl`，10,783 個回合）、所有訓練好的權重、彙整結果（`r1_*.json`、`r3_colony.json`）、逐蟻群結果（`r2_per_seed.csv`，700 個蟻群）與產量曲線（`r2_delivery_curves.csv`）。

可由腳本重建的中間檔因為較大（解壓後約 160 MB），不納入 Git 追蹤：每個方法、每個種子的逐回合預測（`results/la2/r1/*.pkl`）、蟻群的原始紀錄（`results/la2/r2_jobs/*.pkl`），以及雜訊教師語料（`results/la2/corpus_f*.pkl`）。它們以 `learnable-ants-state-tokens-large-data.zip`（5 MB）附在 GitHub release，並一同存檔於 Zenodo（見「引用」）；在儲存庫根目錄解壓即可還原，腳本會直接使用而不重算。

## 授權

- 程式碼（`miat_ants/`（`weights/` 除外）、`experiments/`、`tests/`、`run_all.py`）：[MIT 授權](LICENSE)。
- 資料、圖表、模型權重與文件（`results/`、`figures/`、`miat_ants/la2/weights/`、`miat_ants/foraging/weights/`、`docs/`、README 檔）：[創用 CC 姓名標示 4.0 國際授權（CC BY 4.0）](LICENSE-DATA)。

## 引用

如使用本程式或資料，請引用存檔的版本（DOI 會在第一次 Zenodo 發布後補上）：

> Chen, C.-H. (2026). *Learnable ants: Self-generated state tokens as memory (code and data)* (v1.0.0). Zenodo. https://doi.org/[待補]

機器可讀的引用資訊見 [CITATION.cff](CITATION.cff)。關於群體層級協調的姊妹研究見 [learnable-ants-cwc](https://github.com/PierreChingHanChen/learnable-ants-cwc)。

## 聯絡

問題與錯誤回報：請開 issue。
