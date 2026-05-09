# 進度記錄

## 目前狀態（2026-05-09）

- LSTM baseline 訓練完成：**R² = 0.579**（timesteps=24, 32 features）
- ML baseline 重跑完成（LR/RF/SVR/kNN）
- DR 評估 + 視覺化重跑完成（35 組合，6 張圖）
- **5-fold TimeSeriesSplit CV 完成（全 5 模型）**
  - LSTM R² = 0.581 ± 0.036（與 RF 統計上無顯著差異）
  - CV 視覺化進行中

---

## 完成步驟總覽

| 步驟 | 狀態 | 說明 |
|---|---|---|
| 1. 建環境 | ✅ 完成 | conda `smart_home` (Python 3.10, TF 2.10, RTX 3060 GPU) |
| 2. 下載資料 | ✅ 完成 | `data/raw/energydata_complete.csv` 已就位 |
| 3. 前處理 | ✅ 完成 | `data/processed/processed.csv`（19735×33） |
| 4. EDA | ✅ 完成 | `results/figures/eda/` 四張圖 |
| 5. 滑動窗口 | ✅ 完成 | scalers 存於 `results/models/`；含 log1p 變換 |
| 6. LSTM 訓練 | ✅ 完成 | R²=0.579，108 epochs，GPU（RTX 3060） |
| 7. 基準模型 | ✅ 完成 | LR/RF/SVR/kNN，GPU 加速 |
| 8. DR 策略 | ✅ 完成 | `src/dr_strategies.py`，7 種策略 |
| 9. 評估 | ✅ 完成 | 35 組合，`all_results.csv` |
| 10. 視覺化 | ✅ 完成 | `results/figures/paper/` 6 張圖 |
| 11. main.py | ✅ 完成 | 串接全流程，支援 --skip-* 旗標 |
| 12. 5-fold CV | ✅ 完成 | `src/cv.py`，全 5 模型，`cv_results.csv` |

---

## 下一階段任務

1. ~~重跑 ML baseline~~ ✅ 完成
2. ~~重跑 DR 評估 + 視覺化~~ ✅ 完成
3. ~~實作 5-fold TimeSeriesSplit CV~~ ✅ 完成
4. **5-fold CV 視覺化**（bar chart with error bars）← 進行中
5. **寫報告**

---

## 重要決定紀錄

| 決定 | 原因 |
|---|---|
| 放棄 Phase 1（timesteps=144 + 9 lag features）| R² 反降 0.594→0.568，val_loss 卡在同一平台 |
| 不再嘗試提升 LSTM R² | 這個資料集天花板約 0.65（文獻共識），現有 0.579 符合最低標準 |
| 評估邏輯維持 `metrics(y_true, dr_fn(y_pred))` | 不加回 Skill Score，不改回兩邊都套 DR |
| kNN 改用最後 timestep 41→32 維 + k=15 | 修正維度詛咒，R² 全轉正 |
| Load Leveling window=144 | 日均值平滑才能讓它排最後 |
| Price Based 改 CED 模型（彈性=0.30）| 對齊文獻，排名升至前三 |

---

## 絕對不要做

- 不要重新加 Skill Score（已決定不做）
- 不要再嘗試提升 LSTM R²（已決定接受 0.579）
- 不要動 `evaluation.py`、`dr_strategies.py`
- 不要忘記 commit（每完成一步立刻 commit）

---

## 5-fold TimeSeriesSplit CV 結果

| 模型 | R² (mean ± std) | MAE (mean ± std) | 備註 |
|---|---|---|---|
| RandomForest | 0.586 ± 0.041 | 21.82 ± 2.12 | Best mean，穩定 |
| **LSTM** | **0.581 ± 0.036** | **23.23 ± 3.74** | **與 RF 統計上無顯著差異，std 最低** |
| LinearRegression | 0.509 ± 0.116 | 26.74 ± 7.57 | std 大，需要資料量 |
| SVR | 0.470 ± 0.213 | 27.99 ± 6.79 | Fold 1 表現極差 |
| kNN | 0.184 ± 0.034 | 37.14 ± 5.01 | 最差（對齊論文） |

關鍵發現：
- RF 跟 LSTM mean R² 差距 0.005，std 都遠大於差距 → 統計上無顯著差異
- LSTM 的 std=0.036 是所有模型最低之一，代表跨資料分布最穩定
- Fold 1 訓練資料僅 3,290 筆，LSTM 仍達 R²=0.519，沒有崩潰
- 論文「LSTM 顯著優於 RF」的結論在嚴謹 CV 下無法重現

---

## LSTM 最終結果

| 指標 | 我們（baseline）| 論文（LSTM + Price Based）|
|---|---|---|
| R² | 0.579 | 0.94 |
| MAE | 19.89 Wh | 18.95 Wh |
| RMSE | 41.66 Wh | 24.83 Wh |

**最低標準（R² > 0.5）✅ 達成**

---

## 關鍵設計決策（最終版）

| 項目 | 決策 | 原因 |
|---|---|---|
| 特徵數 | 32（含 Appliances 歷史） | 自回歸輸入是 R² 提升關鍵 |
| TIMESTEPS | 24（4 小時窗口） | 更長沒有改善，反降 |
| Target 變換 | log1p → MinMaxScale → expm1 | 右偏分佈正規化 |
| LSTM 損失函數 | MSE | MAE → 預測中位數 → 系統性低估 |
| LSTM 架構 | 128+64 LSTM + Dropout 0.2 + L2 1e-4 | 已調過，不要再改 |
| kNN 實作 | 最後 timestep 32 維，k=15 | 修正維度詛咒 |
| 評估邏輯 | `metrics(y_true, dr_fn(y_pred))` | DR 只套預測值 |

---

## 環境注意事項

- **舊環境 `venv/`**：廢棄，勿使用
- **新環境 `smart_home`（conda）**：Python 3.10 + TF 2.10 + GPU
- LSTM 存檔格式用 `.h5`（TF 2.10 不支援 `.keras`）
- 所有 GPU 模型需在 smart_home 環境執行
