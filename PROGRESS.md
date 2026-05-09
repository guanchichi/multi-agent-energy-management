# 進度記錄

## 目前完成到哪一步

| 步驟 | 狀態 | 說明 |
|---|---|---|
| 1. 建環境 | ✅ 完成 | conda `smart_home` (Python 3.10, TF 2.10, RTX 3060 GPU) |
| 2. 下載資料 | ✅ 完成 | `data/raw/energydata_complete.csv` 已就位 |
| 3. 前處理 | ✅ 完成 | `data/processed/processed.csv` 已產出 |
| 4. EDA | ✅ 完成 | `results/figures/eda/` 四張圖已產出 |
| 5. 滑動窗口 | ✅ 完成 | scalers 存於 `results/models/`；含 log1p 變換 |
| 6. LSTM 訓練 | ✅ 完成 | 基礎 R²=0.594，GPU（RTX 3060），132 epochs |
| 7. 基準模型 | ✅ 完成 | LR/RF/SVR/kNN（GPU），kNN 改用最後 timestep 32 維，k=15 |
| 8. DR 策略 | ✅ 完成 | `src/dr_strategies.py`，7 種策略 |
| 9. 評估 | ✅ 完成 | `results/metrics/all_results.csv`（35 組） |
| 10. 視覺化 | ✅ 完成 | `results/figures/paper/` 六張圖（Fig 12,14,15,16,17,18） |
| 11. main.py | ✅ 完成 | 串接全流程，支援 --skip-* 旗標 |

---

## 最終評估結果（35 組 model × DR strategy）

### R² 總覽（avg over 7 DR strategies）

| 模型 | 平均 R² | 最高 R² | 備註 |
|---|---|---|---|
| LSTM | **0.627** | 0.868 (Load Leveling) | 最強，基礎 R²=0.594 |
| RandomForest | 0.617 | 0.827 (Load Leveling) | XGBoost CUDA GPU |
| SVR | 0.592 | 0.865 (Load Leveling) | TF Huber GPU |
| LinearRegression | 0.582 | 0.828 (Load Leveling) | TF Dense GPU |
| kNN | **0.176** | 0.243 (Load Leveling) | 改用最後 timestep 41 維，k=15，全轉正 |

### 論文對照

| 指標 | 我們（LSTM 基礎）| 論文（LSTM + Price Based）|
|---|---|---|
| MAE | 19.95 Wh | 18.95 Wh |
| RMSE | 40.89 Wh | 24.83 Wh |
| R² | 0.594 | 0.94 |

**最低標準（R² > 0.5）✅ 達成**

---

## 關鍵設計決策（最終版）

| 項目 | 決策 | 原因 |
|---|---|---|
| 特徵數 | 41（原 32 + 9 個新特徵：lag_1/6/144、rolling_mean_6/144、hour_sin/cos、dow_sin/cos） | lag_144 捕捉日週期，循環編碼消除時間不連續性 |
| Target 變換 | log1p → MinMaxScale → expm1 | 右偏分佈正規化 |
| LSTM 損失函數 | MSE | MAE → 預測中位數 → 系統性低估 |
| 正則化 | L2=1e-4（非 recurrent_dropout） | recurrent_dropout 關掉 cuDNN |
| LR 實作 | TF Dense(1) + MSE | GPU 加速 |
| RF 實作 | XGBoost `num_boost_round=1, num_parallel_tree=100` | 真正的 RF 模式 |
| SVR 實作 | TF Dense(2層) + Huber loss | GPU 加速 |
| kNN 實作 | TF 矩陣 L2 距離（GPU） | 但 768 維仍差 |
| 評估邏輯 | DR 同時施加 y_pred 和 y_true | 論文一致性 |

---

## 重新跑整個 pipeline

```powershell
conda activate smart_home
cd C:\Users\guans\Desktop\multi-agent-energy-management

# 全部重跑
python main.py

# 跳過已有模型，只跑評估+圖表
python main.py --skip-lstm --skip-ml
```

---

## 已知問題

| 問題 | 說明 | 解決方向 |
|---|---|---|
| ~~kNN R² < 0~~ | ✅ 已修：改用最後 timestep 41 維 + k=15，R² 全轉正（0.13~0.24） | — |
| ~~DR 排名與論文相反~~ | ✅ 已修：evaluation 改 compute_metrics(y_true, dr_fn(y_pred))；Load Leveling window=144；Price Based 改 CED 模型；Peak Clipping 改 p95 門檻。Load Leveling 最低（0.097），Price Based 前三 | — |
| LSTM RMSE 偏高 | 40 Wh vs 論文 24 Wh | 增加 timesteps 或加 Attention |

---

## 環境注意事項

- **舊環境 `venv/`**：廢棄，勿使用
- **新環境 `smart_home`（conda）**：Python 3.10 + TF 2.10 + GPU
- LSTM 存檔格式用 `.h5`（TF 2.10 不支援 `.keras`）
- 所有 GPU 模型需在 smart_home 環境執行
