# -*- coding: utf-8 -*-
"""X7 信誉评级得分 的消融实验
=============================

背景：`X7_信誉评级得分` 是按违约标签人为构造的一列
（见 `xgboost_train_model.py` 中的 `[1.00]*80 + [0.75]*20 + [0.50]*10 + [0.25]*12`），
它和标签高度共线，相当于把答案提前放进了特征里。带它训练得到的 AUC
不能代表「从发票数据预测违约」的真实能力。

本脚本在同一份数据、同一套切分下做两组 5 折交叉验证：

- **含 X7**：与主流程一致的全部特征；
- **去 X7**：只用发票衍生特征。

输出 `results/ablation_x7.csv`，并在终端打印两组 AUC 的均值与标准差。

用法::

    python src/ablation_x7.py

依赖：pandas / scikit-learn；装了 xgboost 时优先用 XGBoost，
否则退回 sklearn 的 HistGradientBoosting（脚本会在输出里注明实际后端）。
"""

from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path

# joblib 在部分 Windows 环境探测物理核数会失败并打印一长串 traceback，这里直接给定
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "preprocessed_data_problem1_annualized.csv"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LABEL_COL = "Y_是否违约"
ID_COL = "企业代号"
LEAK_COL = "X7_信誉评级得分"


def make_model(seed: int):
    """优先 XGBoost（与主流程一致），不可用时退回 sklearn。"""
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="auc",
            random_state=seed, n_jobs=4,
        ), "xgboost"
    except ImportError:
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=seed,
        ), "sklearn-HistGradientBoosting"


def cross_val_auc(X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, str]:
    """5 折分层交叉验证，返回每折 AUC 与所用后端名。"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs, backend = [], ""
    for train_idx, test_idx in cv.split(X, y):
        model, backend = make_model(seed)
        model.fit(X.iloc[train_idx], y[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    return np.array(aucs), backend


def main() -> None:
    df = pd.read_csv(DATA_FILE)
    y = df[LABEL_COL].to_numpy()
    feature_cols = [c for c in df.columns if c not in (ID_COL, LABEL_COL)]

    print(f"样本数 {len(df)}，违约 {int(y.sum())} 条，特征 {len(feature_cols)} 个")

    results = []
    for name, cols in (
        ("含 X7（与主流程一致）", feature_cols),
        ("去 X7（仅发票衍生特征）", [c for c in feature_cols if c != LEAK_COL]),
    ):
        aucs, backend = cross_val_auc(df[cols], y)
        results.append({
            "实验组": name,
            "特征数": len(cols),
            "AUC均值": round(float(aucs.mean()), 4),
            "AUC标准差": round(float(aucs.std()), 4),
            "AUC最小": round(float(aucs.min()), 4),
            "AUC最大": round(float(aucs.max()), 4),
            "模型后端": backend,
        })
        print(f"{name}: AUC = {aucs.mean():.4f} ± {aucs.std():.4f} （{backend}）")

    drop = results[0]["AUC均值"] - results[1]["AUC均值"]
    print(f"\n去掉 X7 后 AUC 下降 {drop:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "ablation_x7.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"结果已写入 {out}")


if __name__ == "__main__":
    main()
