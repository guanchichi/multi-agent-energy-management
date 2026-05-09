# 復刻論文實驗規劃書

**論文標題**:AI-driven Optimization of Energy Consumption and Demand Response in Smart Homes (Durrani et al., 2025)

**復刻範圍**:LSTM 預測模型 + 7 種 DR 策略 + 4 個 ML 基準模型(LR / RF / SVR / kNN)

**執行環境**:本地端 Python

---

## 目錄

1. [專案概覽](#1-專案概覽)
2. [環境準備](#2-環境準備)
3. [專案目錄結構](#3-專案目錄結構)
4. [實作步驟拆解](#4-實作步驟拆解)
5. [關鍵技術決策說明](#5-關鍵技術決策說明)
6. [預期結果與驗收標準](#6-預期結果與驗收標準)
7. [常見問題排解](#7-常見問題排解)
8. [延伸方向](#8-延伸方向)

---

## 1. 專案概覽

### 1.1 論文核心方法

論文使用 UCI Appliances Energy Prediction 資料集(19,735 筆 × 29 欄,每 10 分鐘採樣),建立 5 種預測模型,並在每個模型的預測結果上模擬 7 種 Demand Response 策略,最後用 MAE / RMSE / R² 三個指標比較。

### 1.2 論文的目標數字(LSTM)

| DR 策略 | MAE (Wh) | RMSE (Wh) | R² |
|---|---|---|---|
| Peak Clipping | 20.33 | 26.84 | 0.91 |
| Valley Filling | 20.17 | 26.12 | 0.92 |
| Load Shifting | 19.84 | 25.59 | 0.93 |
| Load Leveling | 21.47 | 28.76 | 0.89 |
| ToU Optimization | 19.78 | 25.21 | 0.93 |
| **Price Based** | **18.95** | **24.83** | **0.94** |
| Behavioral DR | 19.67 | 25.47 | 0.93 |

### 1.3 復刻的不確定性

論文有以下細節沒交代清楚,需要自己合理化(這是本次規劃的重點):

- LSTM 的層數、units、timesteps、epochs
- 7 種 DR 策略的數學公式
- Train / Test 切分比例
- Outlier 偵測的具體閾值
- 5-fold CV 的具體做法

→ 因此**復刻不是要 100% 對到論文數字**,而是**復刻出相同的趨勢**(LSTM > RF > 其他、Price-Based 最佳、Load Leveling 最差)。

---

## 2. 環境準備

### 2.1 硬體建議

| 配置 | 最低 | 建議 |
|---|---|---|
| RAM | 8 GB | 16 GB |
| GPU | 不需要 | NVIDIA GPU(可省 5–10 倍訓練時間) |
| 硬碟 | 1 GB | — |

**沒有 GPU 也可以跑**:LSTM 在 CPU 上訓練約 10–20 分鐘,可以接受。

### 2.2 Python 版本

建議 **Python 3.10 或 3.11**(TensorFlow 對 3.12+ 支援還不穩)。

```bash
python --version
# 應該看到 Python 3.10.x 或 3.11.x
```

### 2.3 建立虛擬環境

**強烈建議用 venv 或 conda 隔離環境**,避免汙染全局 Python。

#### 方法 A:使用 venv(內建,推薦)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 方法 B:使用 conda

```bash
conda create -n smart_home python=3.11
conda activate smart_home
```

### 2.4 安裝套件

建立 `requirements.txt`,內容如下:

```text
numpy>=1.24
pandas>=2.0
scipy>=1.11
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
tensorflow>=2.15
jupyter>=1.0
ipykernel>=6.25
```

安裝:

```bash
pip install -r requirements.txt
```

#### 如果有 NVIDIA GPU(選用)

TensorFlow 2.15+ 在 Windows 不再原生支援 GPU,需要透過 WSL2。  
若用 Linux 或 WSL2:

```bash
pip install tensorflow[and-cuda]
```

驗證 GPU 可用:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# 應該看到至少一個 GPU
```

如果跑不起來不用焦慮,**用 CPU 也完全可以**。

### 2.5 資料下載

從 UCI ML Repository 下載資料集:

- **連結**:https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- **下載檔案**:`appliances+energy+prediction.zip`
- **解壓後檔名**:`energydata_complete.csv`(約 12 MB)

把這個檔案放到專案的 `data/raw/` 底下。

---

## 3. 專案目錄結構

```
smart_home_dr/
├── README.md                       ← 簡短說明 + 怎麼跑
├── requirements.txt                ← 套件清單
├── plan.md                         ← 本規劃書
│
├── data/
│   ├── raw/
│   │   └── energydata_complete.csv ← 下載的原始資料
│   └── processed/
│       └── (前處理後的中間檔案)
│
├── src/                            ← 模組化程式碼
│   ├── __init__.py
│   ├── config.py                   ← 全局參數(timesteps, batch_size, paths)
│   ├── preprocessing.py            ← 階段 1:前處理 + 特徵工程
│   ├── eda.py                      ← 階段 2:EDA 圖表
│   ├── sequences.py                ← 階段 3:滑動窗口
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py           ← 階段 4:LSTM 訓練
│   │   └── ml_models.py            ← 階段 5:LR/RF/SVR/kNN
│   ├── dr_strategies.py            ← 階段 6:7 種 DR 策略函數
│   ├── evaluation.py               ← 階段 7:MAE/RMSE/R² 計算
│   └── visualization.py            ← 階段 8:熱圖 + bar plot
│
├── notebooks/                      ← 探索用 Jupyter 筆記本
│   ├── 01_eda.ipynb
│   ├── 02_lstm_training.ipynb
│   └── 03_dr_simulation.ipynb
│
├── results/
│   ├── figures/                    ← 輸出的所有圖
│   ├── models/                     ← 訓練好的模型 (.keras 檔)
│   └── metrics/                    ← 結果 CSV
│
└── main.py                         ← 一鍵執行整個流程
```

### 為什麼這樣切?

- **`src/` 模組化**:每個階段一個檔案,debug 跟修改都容易
- **`notebooks/` 用來探索**:寫程式碼前先在 notebook 試,確認可行再搬到 `src/`
- **`config.py` 集中參數**:超參數調整時只改一個地方
- **`main.py` 一鍵跑**:示範流程怎麼串

---

## 4. 實作步驟拆解

### 步驟 1:前處理(`src/preprocessing.py`)

**輸入**:`data/raw/energydata_complete.csv`  
**輸出**:`data/processed/processed.csv`

#### 1.1 載入資料

- 用 `pd.read_csv()` 讀取
- 把 `date` 欄位轉成 `datetime`
- 按時間排序

#### 1.2 時間特徵展開

從 `date` 拆出:
- `hour`(0–23)
- `day_of_week`(0–6)
- `is_weekend`(0/1)
- `month`(1–12)

#### 1.3 衍生氣象特徵

- **Discomfort Index**(Thom's index):  
  `DI = T_out − 0.55 × (1 − 0.01 × RH_out) × (T_out − 14.5)`
- **Apparent Temperature**:  
  先算水汽壓 `e = (RH_out / 100) × 6.105 × exp(17.27 × T_out / (237.7 + T_out))`,  
  再算 `AT = T_out + 0.33 × e − 0.7 × Windspeed − 4.0`

#### 1.4 缺失值處理

- 用 `df.ffill().bfill()`(這個資料集其實很乾淨,但保險起見)

#### 1.5 Outlier 平滑

- 對 `Appliances` 計算 z-score
- |z| > 3 的點,用前後 ±3 點的中位數取代(保留時序連續性)

#### 1.6 移除欄位

- 丟掉 `rv1`, `rv2`(隨機噪聲,論文有提到)

#### 1.7 存檔

存成 `data/processed/processed.csv`,後面所有步驟都從這個檔讀。

---

### 步驟 2:EDA(`src/eda.py`)

**目的**:再現論文 Figure 2, 3, 4。

- 時間序列折線圖(Appliances over time)
- 直方圖(Distribution of Appliances)
- 相關性熱圖(Pearson correlation matrix)
- 描述統計表(對應論文 Table 2)

存到 `results/figures/eda/`。

---

### 步驟 3:建立滑動窗口(`src/sequences.py`)

LSTM 需要 `(samples, timesteps, features)` 形狀的輸入。

#### 3.1 切 train / test

- 按時間順序切,**不能 shuffle**
- 70% train / 30% test
- 也可以再切出 15% validation

#### 3.2 Min-Max scaling

關鍵點:**scaler 只能 fit 在 train**,再 transform train + test,否則會 data leakage。

```python
feature_scaler = MinMaxScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)  # 注意:用 transform 不是 fit_transform
```

#### 3.3 滑動窗口

```python
def make_sequences(X, y, timesteps=24):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
        ys.append(y[i+timesteps])
    return np.array(Xs), np.array(ys)
```

`timesteps = 24` 表示用過去 4 小時(24 × 10 分鐘)預測下一個 10 分鐘。

---

### 步驟 4:LSTM 訓練(`src/models/lstm_model.py`)

#### 架構

```
Input (24, ~30)
  ↓
LSTM(64, return_sequences=True)
  ↓
Dropout(0.2)
  ↓
LSTM(32)
  ↓
Dropout(0.2)
  ↓
Dense(16, relu)
  ↓
Dense(1)
```

#### 訓練設定

| 參數 | 值 |
|---|---|
| Loss | MSE |
| Optimizer | Adam (lr=0.001) |
| Batch size | 64 |
| Epochs | 50(配 EarlyStopping patience=10) |
| Validation split | 0.15 |

#### 預測 + 反 scaling

```python
y_pred_scaled = model.predict(X_test_seq)
y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()  # 變回 Wh
```

#### 存檔

- 模型存到 `results/models/lstm.keras`
- 訓練歷史存到 `results/metrics/lstm_history.csv`

---

### 步驟 5:基準模型(`src/models/ml_models.py`)

跑 4 個傳統 ML 模型作對照:

| 模型 | 設定 |
|---|---|
| Linear Regression | 預設 |
| Random Forest | n_estimators=50 |
| SVR | kernel='rbf', C=1.0(訓練慢,可 sample 5000 筆) |
| kNN | n_neighbors=5 |

**注意**:傳統 ML 不需要 timesteps,把 LSTM 用的 sequence flatten 成 `(samples, 24 × features)`。

```python
X_train_flat = X_train_seq.reshape(X_train_seq.shape[0], -1)
```

SVR / kNN 在 13,800 筆 × 720 維上訓練很慢,可以隨機 sample 5000 筆訓練。RF 跟 LR 用全部資料。

---

### 步驟 6:DR 策略(`src/dr_strategies.py`)

論文沒給公式,以下是合理化的版本。所有函數的 signature 都是 `fn(y_pred, hours) → y_dr`。

| 策略 | 我的定義 |
|---|---|
| Peak Clipping | Peak hours (18–21) 中超過 P95 的點 → 設為 P95 |
| Valley Filling | Off-peak (0–5) 中低於 P25 的點 → 設為 P25 |
| Load Shifting | Peak 超過 P90 的部分能量 × 0.3,平均加到 off-peak |
| Load Leveling | 對整個序列做 rolling mean,window=6(=1 小時) |
| ToU Optimization | Peak × 0.85,Off-peak × 1.10 |
| Price-Based | `y_new = y × (price/price_avg)^(-0.3)`,price = {peak: 1.5, off: 0.6, mid: 1.0} |
| Behavioral DR | 隨機選 20% 的 peak 點,降 25% |

這些參數可調(P95、彈性 ε、participation 等),調整後可以看出對結果的影響。

---

### 步驟 7:評估(`src/evaluation.py`)

對每個 (Model × DR Strategy) 組合計算:

```python
mae = mean_absolute_error(y_true, y_dr)
rmse = np.sqrt(mean_squared_error(y_true, y_dr))
r2 = r2_score(y_true, y_dr)
```

5 模型 × 7 策略 = 35 組,結果存成 long-format DataFrame:

```
| Model | DR Strategy | MAE | RMSE | R2 |
|-------|-------------|-----|------|-----|
| LSTM  | Peak Clipping | 20.3 | 26.8 | 0.91 |
| LSTM  | Valley Filling | ... | ... | ... |
| ...
```

存成 `results/metrics/all_results.csv`。

---

### 步驟 8:視覺化(`src/visualization.py`)

復刻論文這幾張關鍵圖:

| 論文 Figure | 內容 | 要產出的檔 |
|---|---|---|
| Figure 12 | MAE 熱圖(模型 × DR) | `mae_heatmap.png` |
| Figure 15 | RMSE 熱圖 | `rmse_heatmap.png` |
| Figure 16 | R² 熱圖 | `r2_heatmap.png` |
| Figure 17 | Grouped bar plot of MAE/RMSE | `bar_mae_rmse.png` |
| Figure 18 | Grouped bar plot of R² | `bar_r2.png` |
| Figure 14 | LSTM × 7 DR 時序對比(4×2 subplot) | `lstm_timeseries.png` |

統一存到 `results/figures/`。

---

## 5. 關鍵技術決策說明

### 5.1 為什麼 timesteps = 24?

24 步 × 10 分鐘 = 4 小時。理由:
- 太短(< 12)抓不到 daily pattern 的局部特徵
- 太長(> 96)記憶體吃緊,且訓練變慢
- 24 是經驗上的甜蜜點

如果你的 R² 卡在 0.6 以下沒法再上去,**可以試 timesteps = 48 或 96**。

### 5.2 為什麼用 70/30 切而不是 80/20?

論文沒明講。70/30 的好處是 test 集大,評估比較穩定。如果你想要更接近論文的 R² = 0.94,可以試 80/20 或 85/15(訓練資料越多,精度通常越高)。

### 5.3 為什麼 Min-Max 而不是 Standard scaling?

論文明確寫了用 Min-Max。對 LSTM 來說兩種都可以,Min-Max 把值壓在 [0, 1] 對 sigmoid/tanh 啟用函數比較友善。

### 5.4 為什麼 SVR/kNN 要 sample?

完整 train set 約 13,800 筆,每筆 720 維(24 × 30)。SVR 是 O(n²) 訓練,RAM 跟時間都會爆。Sample 5000 筆是工程妥協,對結果影響不大。

### 5.5 DR 策略的閾值怎麼定?

我選的 P95、P25、ε=0.3 都是文獻常見值。**它們會影響數字但不會影響趨勢**。如果你想對齊論文數字,可以微調這些參數。

---

## 6. 預期結果與驗收標準

### 6.1 LSTM Baseline(無 DR,純預測)

| 指標 | 預期範圍 | 論文 LSTM Baseline |
|---|---|---|
| MAE | 20–35 Wh | ~20 |
| RMSE | 28–50 Wh | ~26 |
| R² | 0.50–0.85 | 0.94 |

**為什麼可能比論文差**:論文的 R² = 0.94 在這個資料集上其實偏高,可能用了更長 timesteps、更多 epochs、或不同的 train/test 切法。

### 6.2 復刻成功的最低標準

只要符合以下三點就算成功:

1. ✅ **模型排名正確**:LSTM > RF > SVR ≈ kNN > LR
2. ✅ **DR 策略排名大致正確**:Price-Based / Behavioral / ToU 表現好,Load Leveling 最差
3. ✅ **量級合理**:LSTM 的 MAE 個位數結尾(< 50 Wh),R² > 0.5

如果三項都符合,就說明你的實作是對的。

### 6.3 進階目標(逼近論文數字)

若想拚 LSTM R² > 0.9:

- timesteps 改 48 或 96
- LSTM units 加到 128
- Epochs 拉到 100
- Train/test 改 80/20
- 加 BatchNormalization

---

## 7. 常見問題排解

### 7.1 TensorFlow 安裝失敗

**症狀**:`pip install tensorflow` 報錯

**解法**:
1. 確認 Python 版本是 3.10 或 3.11
2. Windows 升級到 Visual C++ Redistributable
3. 用 `pip install tensorflow-cpu` 跳過 GPU 版本

### 7.2 訓練 NaN

**症狀**:loss 跑到第 2–3 個 epoch 變 NaN

**解法**:
- 檢查 scaler 是否 fit 在 train(只能 fit train,不能 fit 全部)
- 學習率調小:`Adam(lr=0.0005)`
- 加 gradient clipping:`Adam(clipnorm=1.0)`

### 7.3 R² 一直很低(< 0.3)

可能原因:
- timesteps 太短 → 試 48
- 特徵沒 scale → 檢查 `X_train_scaled.max()` 應該 ≈ 1
- y 沒反 scale → 檢查 `y_pred` 是不是還在 [0, 1] 區間
- Train/test split shuffle 了 → 時序資料絕對不能 shuffle

### 7.4 RAM 爆掉

**症狀**:訓練到一半 process killed

**解法**:
- batch_size 從 64 降到 32 或 16
- timesteps 從 24 降到 12
- SVR/kNN 的 sample 從 5000 降到 2000

### 7.5 訓練太慢

CPU 上 LSTM 預期 10–20 分鐘。如果超過 1 小時:
- 確認 epochs 沒設太大(50 就夠)
- batch_size 加到 128
- 用 EarlyStopping 提早結束

### 7.6 DR 策略結果跟論文差很多

正常,因為論文沒給公式。如果你的 Price-Based 反而比 Load Leveling 差,可以:
- 調整 `elasticity` 參數(0.1 ~ 0.5)
- 調整 peak/off-peak 的 hour 範圍
- 檢查 `test_hours` 對齊是否正確(要從 `df_test['hour'][TIMESTEPS:]` 取)

---

## 8. 延伸方向

跑完基本復刻後,可以做的進階探索:

### 8.1 5-fold Time Series CV

論文有提但沒強調。用 `sklearn.model_selection.TimeSeriesSplit`:

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

對 LSTM 比較貴(要訓練 5 次),但可以更穩健地評估。

### 8.2 加入 Attention 機制

論文提到 Wan et al. 的 attention-based CNN-LSTM。可以在現有 LSTM 上加 attention layer,看能否衝上 R² > 0.9。

### 8.3 SHAP 可解釋性分析

論文 Table 1 有提到 SHAP。可以用 `shap` 套件畫出哪些特徵對預測最重要(`T1`、`hour`、`RH_1` 應該是 top 3)。

### 8.4 真實 ToU 電價

可以套用台電的時間電價(尖峰、半尖峰、離峰),讓 Price-Based DR 更貼近現實。

### 8.5 把 7 種 DR 策略可調參數放到 config

例如把 `PEAK_HOURS`、`elasticity`、`participation` 都拉成 config 參數,做 sensitivity analysis 並畫圖。

---

## 9. 開發建議的時間規劃

| 階段 | 預估時數 |
|---|---|
| 環境設定 + 資料下載 | 0.5 hr |
| 步驟 1–2(前處理 + EDA) | 2 hr |
| 步驟 3–4(LSTM 訓練) | 3 hr |
| 步驟 5(基準模型) | 1.5 hr |
| 步驟 6(DR 策略) | 2 hr |
| 步驟 7–8(評估 + 視覺化) | 2 hr |
| 除錯 + 調整 | 3 hr |
| **總計** | **~14 hr** |

如果熟悉 ML,可以壓縮到 6–8 小時。

---

## 10. 下一步建議

按這個順序開始:

1. **先建環境** → 跑 `python -c "import tensorflow"` 確認沒錯誤
2. **下載資料** → 把 `energydata_complete.csv` 放到 `data/raw/`
3. **先在 notebook 試跑前處理** → 確認資料形狀正確
4. **跑 LSTM baseline**(不含 DR)→ 看 R² 量級對不對
5. **加入 DR 策略 + 基準模型** → 完整跑一遍
6. **畫圖 + 寫 main.py 串起來**

每個階段跑完都先 commit / 存檔,方便回頭比對。

---

**準備好就告訴我要從哪一步開始實作,我可以給你對應檔案的完整程式碼。**
