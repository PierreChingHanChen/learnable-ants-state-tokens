# `miat_ants.la2`：44 詞元雙字母語彙的可學習螞蟻

本研究的模組。沿用 `miat_ants.foraging` 的 GRAFCET 系統，替換世界、教師與核心。

| 模組 | 內容 |
|---|---|
| `vocab.py` | 44 詞語彙：25 感知詞、10 行動詞、8 控制狀態詞；bo（o a）、aux（o a，另有狀態讀出頭，不回饋）與 dual（o s a）三種格式；`sample_windows` 的 trip／step／sigma 取樣加權 |
| `world.py` | World2：食物品質、食物地點向量、放棄規則、強弱路徑感知、U/K 行動、食物耗盡與重生、感知翻轉雜訊；地圖 `two_patch`、`double_bridge` 與未見地圖 `barrier` |
| `teacher.py` | EthoTeacher：以八個文獻行為狀態運作的規則式教師 |
| `core.py` | GPTCore2：bo／aux／dual 的 microGPT 螞蟻（dual 先說狀態再行動）；線上自我模仿、取樣加權與教師複習（`rehearse`） |
| `gru.py` | 參數量相同的 GRU（4,979 參數），numpy 手寫完整回合 BPTT；bo 與 aux 兩種目標 |
| `system.py` | `LA2System`／`LA2Config`、`base_path`（訓練好權重的路徑） |
| `pretrain.py` | 教師語料收集、訓練／保留回合切分與行為複製 |
| `metrics.py` | 熵、n-gram、`BayesStateFilter`（IO-HMM 有限狀態估計器）、自由追蹤（多代理人批次）、aux 讀出、探針、分群、因果介入、合法率、標記調節、回合配對 bootstrap |
| `weights/` | 所有訓練好的模型（`.npz`）與其訓練設定（`.json`） |

權重命名：`la2_{bo,aux,dual}_s{k}`（訓練種子 k＝0–9）、`la2_{aux,dual}_ctx{4,8}_s{k}`（上下文消融）、`la2_{bo,dual}_f{05,10,20}_s{k}`（感知雜訊 5%／10%／20%）、`gru_{bo,aux}_s{k}`。`la2_bo`、`la2_dual` 與 `_s0` 相同，保留作為預設權重。

實驗腳本在 `experiments/`，執行順序見根目錄的 `run_all.py` 與 `docs/`。測試：`tests/test_la2.py`。
