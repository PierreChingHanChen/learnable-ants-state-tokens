[English](../en/DATA_DICTIONARY.md) | 中文

# 資料字典

`results/la2/` 中的所有檔案，以及寫出它們的步驟（見 [EXPERIMENTS.md](EXPERIMENTS.md)）。標示 **（封存）** 的檔案不在 Git 儲存庫中，而在大型資料封存 `learnable-ants-state-tokens-large-data.zip`（附在 GitHub release，並存檔於 Zenodo）；在儲存庫根目錄解壓即可放到正確位置。

Pickle 檔（`.pkl`）以 Python 的 `pickle` 模組寫出。請只載入來自本儲存庫或其封存的檔案。

## 詞元序列

一個*回合*（episode）是詞元編號的列表（`miat_ants.la2.vocab.STOI`／`ITOS`）。它以回合起始詞 `.`（編號 0）開頭，接著是 dual 格式 `o s a o s a …`：每一步的感知、教師的控制狀態與行動。螞蟻在巢放下食物（`D`）或走滿 400 步（逾時）時回合結束。bo 與 aux 的序列以 `miat_ants.la2.pretrain.as_mode(trips, "bo")` 取得，它會去掉狀態詞。

| 組別 | 詞元 |
|---|---|
| 行動（10） | `F L R` 前進／左前／右前；`f l r` 同上並沉積費洛蒙；`U` U 形迴轉；`K` 原地掃視；`T` 拿取食物；`D` 在巢放下食物 |
| 感知：搜尋（10） | `X` 腳下優質食物；`Y` 腳下劣質食物；`A B C` 食物在前／左前／右前；`P Q S` 費洛蒙在前／左前／右前；`N` 無線索；`W` 前方是牆 |
| 感知：食物地點向量（5） | `g h j k` 記憶中的食物地點在前／右／後／左；`z` 已到記憶地點但沒有食物 |
| 感知：攜帶（10） | `0` 在巢內；`w` 前方是牆；`1 2 3 4` 巢在前／右／後／左（腳下路徑弱）；`5 6 7 8` 同上（路徑強） |
| 控制狀態（8） | `ε` 探索；`φ` 跟隨路徑；`ν` 向量導航；`σ` 系統搜尋；`κ` 掃視；`υ` U 形迴轉；`η` 攜帶返巢；`δ` 卸貨 |

## 教師語料

| 檔案 | 步驟 | 內容 |
|---|---|---|
| `corpus.pkl` | pretrain | 10,783 個教師回合（two-patch 與 double bridge，模擬種子 500–505）。`miat_ants.la2.pretrain.split(trips, seed=0)` 給出全文使用的 9,705 個訓練回合與 1,078 個保留回合 |
| `corpus_f05.pkl`、`corpus_f10.pkl`、`corpus_f20.pkl` **（封存）** | noise | 同樣的收集方式，感知詞以 0.05、0.1 或 0.2 的機率翻轉 |
| `r1/gen_<env>_eps.pkl` | offline（`gen`） | 各測試環境（`barrier`、`food_fast`、`food_slow`、`decay_fast`、`decay_slow`）與一次新的 two-patch 執行（`train`）的教師回合 |
| `r2_stability_eps.pkl` | colonies | 狀態穩定性檢驗使用的 150 個固定 two-patch 教師回合（模擬種子 900） |

## 離線結果

| 檔案 | 步驟 | 內容 |
|---|---|---|
| `e0_pretrain.json` | pretrain | 種子 0–4 的每個 bo／dual 模型一筆：保留集行動 NLL 與準確率、設定、參數量、訓練時間 |
| `e124.json` | entropy | `e1`：狀態的條件熵（位元；表 S5）與狀態佔比；`ngram`：n = 1–16 的 n-gram 準確率；`seeds`：各種子的自由追蹤、探針、介入與分群純度 |
| `e124_f05.json`、`e124_f10.json`、`e124_f20.json` | noise | 雜訊語料的同樣結果（3 個種子） |
| `e3_summary.csv` | ablation | 由 `e124*.json` 整理，每個雜訊等級與種子一列：`flip`、`seed`、`dual_free`（及 `_sigma`、`_phi`）、`bo_probe`、`obs_probe`、`ngram8`（及 `_sigma`、`_phi`）、`ngram1`、`action_acc_bo`、`action_acc_dual`、`follow` |
| `e5.json`、`e5_jobs/*.json` | ablation | 每個設定、格式與種子一筆 |
| `e5_summary.csv` | ablation | `config`（`ctx4`、`ctx8`、`ctx32`、`embd32`、`data10`、`data30`、`main`）、`mode`、`seed`、`action_nll`、`action_acc`，以及 dual 模型的自由追蹤 `free`、`free_sigma`、`free_phi`（表 S12） |
| `r1_main.json` | offline | `summary`：每個方法的整體（`all`）、各狀態與狀態平衡準確率；學習型方法另有 `min`、`max` 與 `per_seed`。`pairs`：`"<a>\|<b>\|<狀態或 all>"` 的回合配對 bootstrap（平均差、95% 區間與 n）。`eval`：各格式的行動預測。`intervene`：每個種子一筆（`n`、`follow_frac`、`js_orig`、`js_swapped`）。`occupancy_heldout`：保留回合中各狀態的佔比 |
| `r1_perseed.json` | offline | 依訓練種子排列；每個種子 `dual`、`aux`、`gru_aux`、`bo_probe`、`gru_bo_probe`、`rand_probe` 的整體與各狀態準確率（表 S8） |
| `r1_ctx.json` | offline | `rows`：aux 與 dual 模型依上下文長度的追蹤結果；`pairs`：各上下文的 dual − aux（表 S11） |
| `r1_noise.json` | offline | `rows`：各雜訊等級、各方法的追蹤結果；`pairs`：dual 對 n-gram 與濾波器（表 S9） |
| `r1_gen.json` | offline | 各環境的 `summary`、`pairs`、`n_trips`、`occupancy`（表 S20） |
| `r1_act.json` | offline | 每個保留步驟上各格式的行動一致率與 NLL；dual 分別在給定教師狀態與自己說出的狀態下計算（表 S4） |
| `r1/seed_<eval>_<k>.pkl` **（封存）** | offline | 每個訓練種子 k 與評估集：各學習型方法的逐回合狀態預測（`pred`）、行動評估（`eval`）、介入（`intervene`） |
| `r1/det_<eval>.pkl` **（封存）** | offline | n-gram 與有限狀態濾波器的逐回合預測 |
| `r1/ctx_<mode>_<ctx>_<k>.pkl`、`r1/noise_f<p>_<k>.pkl` **（封存）** | offline | 上下文與雜訊部分的逐回合預測 |

方法代碼：`ngram1`（查表）到 `ngram16`、`filter`、`obs_probe`、`rand_probe`、`bo_probe`、`gru_bo_probe`、`aux`、`gru_aux`、`dual`。準確率以比例（0–1）表示。

## 群體結果

| 檔案 | 步驟 | 內容 |
|---|---|---|
| `r2_jobs/<cond>__<env>__<seed>.pkl` **（封存）** | colonies | 每個蟻群一筆：`r2_per_seed.csv` 的各欄，加上 `deliv_curve`（每 50 步的累計運回量） |
| `r2_per_seed.csv` | analysis | 700 列，每個蟻群一列（欄位見下） |
| `r2_delivery_curves.csv` | analysis | 每 50 步的累計運回量：`cond`、`env`、`seed`、`step`（50–3,000）、`delivered`；42,000 列（論文未繪出，供再分析） |
| `r3_colony.json` | analysis | `tables`：各環境、各條件每欄的平均與 95% 區間；`paired`：配對條件之間的產量差（`env`、`a`、`b`、`mean`、`ci_lo`、`ci_hi`、`rel_pct`、`p`、`p_holm`、`n`）；`stab_paired`：穩定性、σ 佔比、練習中的 σ 與逾時的同樣比較 |
| `r3_summary.txt` | analysis | `la2_r3_analysis.py` 印出的摘要 |
| `e7_per_seed.csv` | teacher_search | `teacherS`、`dual_learnA`、`teacher_nomem` 各 20 個 two-patch 蟻群，以及凍結 dual 蟻群 10–19；另有 `sig_in_success`、`sig_in_fail`、`sigfrac_success`、`sigfrac_all`、`n_success_eps`、`n_fail_eps`（表 S19） |

條件（`cond`）：`teacher`；`teacher_nomem`（沒有食物地點記憶的教師）；`bo_frozen`、`aux_frozen`、`dual_frozen`（不學習）；`bo_learn`、`aux_learn`、`dual_learn`（線上自我模仿，依回合取樣）；`dual_learn_step`（依步數取樣）；`dual_learn_sigma`（搜尋平衡取樣）；`dual_learn_reh`（每批四分之一取自教師訓練回合）；`dual_learn_rehNS`、`dual_learn_rehS`（教師複習只用不含／含 σ 的回合）；`dual_learn_A`（dual 只學行動）。

環境（`env`）：`two_patch`、`double_bridge`、`barrier`、`food_fast`、`food_slow`、`decay_fast`、`decay_slow`。

`r2_per_seed.csv` 的欄位：

| 欄位 | 意義 |
|---|---|
| `cond`、`env`、`seed` | 條件、環境、模擬種子 |
| `train_seed` | 起始權重的訓練種子（`seed mod 10`；教師為 −1） |
| `delivered` | 3,000 步內運回巢的食物單位 |
| `takes_rich`、`takes_poor`、`rich_share` | 從優質與劣質食物塊取走的食物，以及優質所佔比例 |
| `gate_short`、`gate_long`、`short_share` | 短橋與長橋的通過次數，以及短橋所佔比例（僅 double bridge） |
| `timeouts`、`success_eps` | 放棄與成功的回合數 |
| `oa_legal`、`osa_legal`、`trans_legal` | 感知→行動、（感知, 狀態）→行動、狀態→狀態的轉移中，出現在教師語料中的比例 |
| `occ_<狀態>` | 說出（教師則為處於）各狀態的步數比例 |
| `mark_q1`、`mark_q0` | 沿路徑攜帶移動時的標記率，優質（q1）與劣質（q0）食物 |
| `mark_q<q>_weak`、`mark_q<q>_strong` | 同上，分弱路徑與強路徑 |
| `mark_q<q>_near`、`mark_q<q>_far` | 拿取食物後 8 步內或之後 |
| `mark_searched<0/1>_q<q>` | 該回合未曾／曾經到過空的記憶地點 |
| `dispersion` | 執行結束時，螞蟻兩兩之間在固定探針上下文上行動分布的平均 Jensen–Shannon 散度 |
| `sig_in_success`、`sig_in_all` | 成功回合、全部回合中含 σ 的比例 |
| `stab_all`、`stab_<狀態>` | 狀態穩定性：以螞蟻最終權重在 150 個固定教師回合上自由追蹤的整體與各狀態準確率（僅 two-patch 的 aux 與 dual） |
| `practiced_sigma`、`practiced_sigma_late` | 線上學習所練習的狀態目標中 σ 的比例，全程與後半段（僅 dual 與 aux 學習） |
| `seconds` | 執行時間 |

空白表示該指標不適用（例如 bo 螞蟻的 `occ_*`，或 two-patch 地圖上的 `short_share`）。

## 權重

`miat_ants/la2/weights/` 存放每個訓練好的模型：`<name>.npz`（參數）與 `<name>.json`（設定、保留集行動 NLL 與準確率、狀態準確率、參數量、訓練時間與訓練曲線）。命名規則見 [`miat_ants/la2/README.md`](../../miat_ants/la2/README.md)。transformer 以 `miat_ants.microgpt.MicroGPT.load(path)` 載入，GRU 以 `miat_ants.la2.gru.GRUModel.load(path)` 載入。`miat_ants/foraging/weights/` 存放姊妹研究的 24 詞元基準模型。
