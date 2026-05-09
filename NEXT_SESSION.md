# 下次開工提示

## 給 Claude 的開場白（直接貼）

```
請先讀 CLAUDE.md 跟 PROGRESS.md，
然後跟我確認：上次做到哪 / 這次要做什麼 / 環境跟設定有沒有要注意的
```

---

## 當前狀態快照（2026-05-09）

- LSTM baseline 已訓練完成：**R² = 0.579**（timesteps=24，32 features）
- 程式碼已回滾到 Phase 1 之前的乾淨狀態
- DR 策略、評估邏輯已定稿

## 下一步任務（按順序）

### Step 1：重跑 ML baseline

```powershell
conda activate smart_home
cd C:\Users\guans\Desktop\multi-agent-energy-management
python main.py --skip-lstm --skip-eval --skip-viz
```

完成後立刻 commit：
```powershell
git add results/metrics/
git commit -m "Retrain ML baseline (LR/RF/SVR/kNN)"
```

### Step 2：重跑 DR 評估 + 視覺化

```powershell
python main.py --skip-lstm --skip-ml
```

完成後立刻 commit：
```powershell
git add results/metrics/all_results.csv results/figures/
git commit -m "Re-run DR evaluation and visualization"
```

### Step 3：實作 5-fold TimeSeriesSplit CV

新增 `src/cv.py`，使用 `sklearn.model_selection.TimeSeriesSplit(n_splits=5)`。
這是論文方法論最後一個缺項。

### Step 4：寫報告

---

## 絕對不要做

- 不要重新加 Skill Score
- 不要再嘗試提升 LSTM R²
- 不要動 `evaluation.py`、`dr_strategies.py`（參數已定論）
- 不要讓 Claude 直接執行 git / python 指令（只給指令讓使用者跑）
