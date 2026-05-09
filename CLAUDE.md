# CLAUDE.md — 專案快速上手

## 專案目標

復刻論文 **Durrani et al. (2025)「AI-driven Optimization of Energy Consumption and Demand Response in Smart Homes」**。

核心工作：
1. 用 UCI Appliances Energy Prediction 資料集訓練 LSTM + 4 種 ML 基準模型
2. 在預測結果上模擬 7 種 Demand Response (DR) 策略
3. 用 MAE / RMSE / R² 比較所有組合，復刻論文圖表

復刻目標是趨勢正確（LSTM > RF > 其他、Price-Based 最佳、Load Leveling 最差），不是完全對齊數字。

---

## 重要文件位置

| 文件 | 路徑 |
|---|---|
| 完整規劃書 | `plan.md` |
| 論文 PDF | `docs/durrani-et-al-2025-ai-driven-optimization-of-energy-consumption-and-demand-response-in-smart-homes.pdf` |
| 進度記錄 | `PROGRESS.md` |

---

## 執行環境

**使用 conda 環境 `smart_home`，不要用根目錄的 `venv/`（舊的、無 GPU）。**

```powershell
conda activate smart_home
```

| 項目 | 值 |
|---|---|
| Python | 3.10 |
| TensorFlow | 2.10.0（最後一個原生 Windows GPU 版） |
| GPU | RTX 3060 Laptop（已驗證可用） |
| cudatoolkit | 11.2 |
| cuDNN | 8.1 |
| Python 路徑 | `C:\Users\guans\anaconda3\envs\smart_home\python.exe` |

---

## 目錄結構

```
multi-agent-energy-management/
├── CLAUDE.md               ← 本文件
├── PROGRESS.md             ← 進度與待辦
├── plan.md                 ← 完整規劃書（步驟、超參數、驗收標準）
├── requirements.txt        ← pip 套件清單（numpy<2 限制）
│
├── docs/
│   └── durrani-et-al-2025-...pdf  ← 論文
│
├── data/
│   ├── raw/
│   │   └── energydata_complete.csv   ← UCI 原始資料（19735×29）
│   └── processed/
│       └── processed.csv             ← 前處理後（19735×33）
│
├── src/
│   ├── config.py           ← 全局參數（路徑、超參數、PEAK_HOURS）
│   ├── preprocessing.py    ← 前處理（時間特徵、氣象特徵、outlier 平滑）
│   ├── eda.py              ← EDA 圖表
│   ├── sequences.py        ← 滑動窗口 + MinMaxScaler
│   └── models/
│       ├── lstm_model.py   ← LSTM 訓練（存 .h5 格式）
│       └── ml_models.py    ← LR / RF / SVR / kNN（待寫）
│
├── results/
│   ├── figures/eda/        ← EDA 圖（timeseries, distribution, hourly, heatmap）
│   ├── models/             ← lstm.h5, feature_scaler.pkl, target_scaler.pkl
│   └── metrics/            ← lstm_history.csv, predictions, all_results.csv
│
└── venv/                   ← 廢棄，勿使用
```

---

## 目前進度

**已完成**：前處理 → EDA → 滑動窗口 → LSTM 程式碼

**下一步**：在 `smart_home` 環境跑 LSTM 訓練

```powershell
conda activate smart_home
cd C:\Users\guans\Desktop\multi-agent-energy-management
python -m src.models.lstm_model
```

詳細待辦清單見 `PROGRESS.md`。

---

## 關鍵設計決策

- **timesteps = 24**：4 小時歷史窗口（24 × 10 分鐘）
- **train/test = 70/30**，按時間順序切，不 shuffle
- **scaler 只 fit train**，避免 data leakage
- **LSTM 存檔用 `.h5`**（TF 2.10 不支援 `.keras`）
- **SVR/kNN 訓練取樣 5000 筆**（避免 RAM 爆掉）
- DR 策略的公式是合理化版本（論文未提供），參數在 `src/config.py` 可調

---

## 論文目標數字（LSTM）

| DR 策略 | MAE | RMSE | R² |
|---|---|---|---|
| Peak Clipping | 20.33 | 26.84 | 0.91 |
| Valley Filling | 20.17 | 26.12 | 0.92 |
| Load Shifting | 19.84 | 25.59 | 0.93 |
| Load Leveling | 21.47 | 28.76 | 0.89 |
| ToU Optimization | 19.78 | 25.21 | 0.93 |
| **Price Based** | **18.95** | **24.83** | **0.94** |
| Behavioral DR | 19.67 | 25.47 | 0.93 |

復刻成功最低標準：模型排名正確 + DR 排名大致正確 + LSTM R² > 0.5。
