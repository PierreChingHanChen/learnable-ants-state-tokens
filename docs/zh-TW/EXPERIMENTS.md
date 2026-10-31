[English](../en/EXPERIMENTS.md) | 中文

# 實驗指南

本指南逐一說明 `run_all.py` 的每個步驟：做什麼、用什麼設定、輸出到哪裡。節號指論文的章節，「表 S…」指補充材料的表格。所有腳本都在 `experiments/`，並從該目錄執行（`run_all.py` 會自動處理）；除非另外註明，輸出都寫到 `results/la2/`。

每個步驟都會略過輸出已存在的工作：`miat_ants/la2/weights/` 的訓練好權重、`results/la2/r1/` 的逐回合預測、`results/la2/r2_jobs/` 的蟻群紀錄、`results/la2/e5_jobs/` 的消融工作。要重算某項，先刪除它的輸出。平行步驟預設使用兩個工作行程（群體步驟可用 `LA2_WORKERS` 調整）。

## 共同設定

| 設定 | 值 |
|---|---|
| 語彙 | 44 個詞元：25 個感知詞、10 個行動詞、8 個控制狀態詞，以及回合起始詞（`miat_ants/la2/vocab.py`） |
| 格式 | **bo** `. o a o a …`；**aux** 為 bo 序列，另在感知位置加一個狀態讀出頭（不回饋）；**dual** `. o s a o s a …` |
| 教師語料 | 教師蟻群在 two-patch 與 double-bridge 地圖上，模擬種子 500–505，各 1,500 步：共 10,783 個回合，切分為 9,705 個訓練回合與 1,078 個保留回合（`pretrain.split`，種子 0） |
| Transformer | numpy 版 microGPT：嵌入寬度 16、上下文 16 步、約 5,000 參數（bo 與 aux 4,992；dual 5,248） |
| Transformer 訓練 | 3,000 個 Adam 步驟、批次 64、學習率 10⁻² 線性遞減 |
| GRU | 隱藏維度 25、4,979 參數；以完整回合做時間反向傳播；3,000 個 Adam 步驟、每步 64 個回合、學習率 10⁻² 線性遞減、梯度截斷 1 |
| 訓練種子 | 每種格式 10 個（0–9）；上下文長度消融與雜訊實驗每格 3 個 |
| 蟻群 | 30 隻螞蟻、共享費洛蒙介質（ρ = 0.97）、3,000 步；模擬種子 k 使用訓練種子 k mod 10 的權重 |
| 線上學習 | 每隻螞蟻保留最近 20 個成功回合；每 50 步做 4 個 Adam 步驟（學習率 3·10⁻³、批次 16） |
| 統計 | 95% 百分位 bootstrap（10,000 次重抽）；離線比較以回合為單位配對 bootstrap；蟻群產量以 Wilcoxon 符號等級檢定並做 Holm 校正 |

下列執行時間以兩核心 CPU 量測（Python 3.13、numpy 2.5）。

## 步驟

### `pretrain`：教師語料與第一批模型（`la2_e0_pretrain.py`）
若 `corpus.pkl` 不存在則收集教師語料，並訓練訓練種子 0–4 的 bo 與 dual 模型（`LA2_SEEDS`，預設 5）。權重：`miat_ants/la2/weights/la2_{bo,dual}_s{0..4}`。摘要：`e0_pretrain.json`。每個模型約 2 分鐘。

### `train`：其餘種子與比較模型（`la2_r0_train.py`）
訓練 bo 與 dual 種子 5–9、aux 種子 0–9、上下文長度消融（aux 與 dual 的 4 步與 8 步上下文，種子 0–2），以及 GRU-bo 與 GRU-aux 種子 0–9（`LA2_NSEEDS`，預設 10）。約 52 個模型、每個 2–3 分鐘，合計約 2 個核心小時。

### `noise`：感知雜訊（`la2_e3_noise.py`，接著 `la2_e124_offline.py`）
對每個翻轉機率 p ∈ {0.05, 0.1, 0.2}（`LA2_FLIPS`），以「感知詞以機率 p 換成同組的隨機詞」重新收集教師語料（`corpus_f05.pkl` 等；教師依雜訊後的詞行動），並訓練 bo 與 dual 模型種子 0–2（`la2_{bo,dual}_f{05,10,20}_s{k}`）。接著對這些模型做第一次離線評估（`e124_f{05,10,20}.json`）。論文使用的配對評估（圖 3C、表 S9）屬於 `offline` 步驟。

### `entropy`：狀態熵與分群（`la2_e124_offline.py`）
教師狀態的條件熵（4.1 節、表 S5：`e124.json` 的 `e1`）、n-gram 基準，以及 bo 激活的分群純度（4.6 節），訓練種子 0–4。

### `ablation`：其他消融（`la2_e5_ablation.py`，接著 `la2_e35_summary.py`）
上下文 4、8、32 步，嵌入寬度 32，以及只用 10% 或 30% 的訓練回合；bo 與 dual 各種子 0–2（表 S12）。工作快取在 `e5_jobs/`；結果在 `e5.json` 與 `e5_summary.csv`。`la2_e35_summary.py` 也由 `e124*.json` 寫出 `e3_summary.csv`，並以 Markdown 印出兩份摘要。

### `offline`：配對離線評估（`la2_r1_offline.py`）
所有方法都在同樣的 1,078 個保留回合（54,523 步）上評估，並逐回合配對比較。`LA2_R1_PART` 選擇要跑的部分：

| 部分 | 內容 | 輸出 | 論文 |
|---|---|---|---|
| `main` | 查表與 n-gram（n = 1、2、4、8、16）、有限狀態濾波器、探針（bo、GRU-bo、隨機初始化 transformer、當步感知）、aux、GRU-aux、dual 自由追蹤；介入檢驗（每個種子 4,000 例） | `r1_main.json`、`r1_perseed.json`、`r1/seed_heldout_*.pkl`、`r1/det_heldout.pkl` | 圖 2；表 S6–S8、S10 |
| `ctx` | aux 與 dual 的 4、8、16 步上下文，種子 0–2 | `r1_ctx.json`、`r1/ctx_*.pkl` | 圖 3A–B；表 S11 |
| `noise` | dual、n-gram 與濾波器在雜訊語料上 | `r1_noise.json`、`r1/noise_*.pkl` | 圖 3C；表 S9 |
| `gen` | 所有方法在五個測試環境及一次新的 two-patch 執行的教師回合上 | `r1_gen.json`、`r1/gen_*_eps.pkl`、`r1/seed_<env>_*.pkl` | 圖 5A；表 S20 |
| `act` | 每個保留步驟上的 top-1 行動一致率與負對數概似 | `r1_act.json` | 4.1 節；表 S4 |

`r1/*.pkl` 為逐回合預測（合計約 150 MB）。本儲存庫只保留 `gen_*_eps.pkl` 回合集，其餘放在大型資料封存。

### `colonies`：群體實驗（`la2_r2_colony.py`）
30 隻螞蟻的閉環蟻群，3,000 步。`LA2_R2_PLAN` 列出工作清單，以分號分隔：

| 清單 | 條件 | 環境 | 蟻群數 |
|---|---|---|---|
| `main` | teacher、teacher_nomem、{bo, aux, dual}_{frozen, learn}、dual_learn_step、dual_learn_sigma、dual_learn_reh、dual_learn_A | two-patch | 12 × 20 |
| `reh_split` | dual_learn_rehNS（教師複習不含 σ 回合）、dual_learn_rehS（只用含 σ 回合） | two-patch | 2 × 20 |
| `bridge` | teacher、{bo, aux, dual}_{frozen, learn} | double bridge | 7 × 20 |
| `gen` | teacher、{bo, aux, dual}_frozen | barrier、food fast、food slow、decay fast、decay slow | 4 × 5 × 10 |
| `gen_learn` | {bo, aux, dual}_learn、dual_learn_reh | food fast、food slow | 4 × 2 × 10 |

合計 700 個蟻群，約 10 個核心小時（凍結蟻群 20–30 秒、學習中的 dual 蟻群約 100 秒、教師蟻群約 10 秒）。在 aux 與 dual 螞蟻的 two-patch 執行中，狀態穩定性檢驗以每隻螞蟻的最終權重，在 150 個固定教師回合上做自由追蹤（`r2_stability_eps.pkl`，模擬種子 900）。每個蟻群寫一筆紀錄到 `r2_jobs/<cond>__<env>__<seed>.pkl`。

測試環境：*barrier* 是未見過的 36 × 36 地圖，有兩道內牆與三個食物塊；*food fast* 與 *food slow* 是每格食物 10 或 90 單位（原為 30）的 two-patch 地圖；*decay fast* 與 *decay slow* 的費洛蒙保留率 ρ 為 0.94 或 0.99（原為 0.97）。

### `teacher_search`：教師回合中的搜尋（`la2_e7_followup.py`）
重跑 20 個 two-patch 教師蟻群，記錄系統搜尋（σ）在成功與失敗回合中出現的頻率，並附兩個較早的條件（dual_learnA、teacher_nomem）與 10 個額外的凍結 dual 蟻群（表 S19）。使用 `la2_e6_colony.py` 中的蟻群函式。輸出：`e7_per_seed.csv`。

### `analysis`：群體統計（`la2_r3_analysis.py`）
以模擬種子配對各條件，計算 bootstrap 區間與 Wilcoxon 檢定，並在每個環境內做 Holm 校正。輸出：`r3_colony.json`；印出的摘要存為 `r3_summary.txt`。若原始蟻群紀錄 `r2_jobs/*.pkl` 存在，也會重寫 `r2_per_seed.csv` 與 `r2_delivery_curves.csv`；否則改讀 `r2_per_seed.csv`，因此只用本儲存庫即可重算統計。

### `figures`：圖 2–5（`la2_r4_figures.py`）
讀取 `r1_main.json`、`r1_ctx.json`、`r1_noise.json`、`r1_gen.json` 與 `r2_per_seed.csv`，寫出 `figures/paper/fig{2..5}_*.{pdf,png}`（設定 `FIGDIR` 可改寫到他處）。圖 1 為示意圖，直接提供。

### `tables`：補充表格（`la2_r5_supp_tables.py`）
把 LaTeX 表 S4、S6–S11、S13–S21 寫到 `results/la2/supp_tables/`（設定 `TABDIR` 可改寫到他處），內容與補充材料中的表格完全相同。表 S1–S3（語彙、教師與模型設定）、S5（熵，取自 `e124.json`）與 S12（消融，取自 `e5_summary.csv`）為手動排版。

## 測試

`python -m pytest -q` 執行 14 個測試（`tests/test_la2.py`，約 10 秒）：語彙與三種格式、教師序列的合法性、食物地點記憶與放棄、U 形迴轉與掃視、食物地點向量、短的 dual 蟻群、n-gram 與熵、aux 目標的對齊、搜尋平衡取樣、快取的前向計算、GRU 梯度與有限差分比對、有限狀態濾波器、會記錄但不回饋狀態的 aux 蟻群，以及 barrier 地圖上的食物重生。
