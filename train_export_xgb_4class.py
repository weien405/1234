#!/usr/bin/env python
# coding: utf-8
"""
===========================================================
四分类 XGBoost + SMOTE + BayesSearchCV + 评估 + SHAP合并图
并额外导出 Streamlit App 需要的两份文件：
- X_test.csv
- XGB.pkl（模型bundle：含 best_model、feature_cols、train_median、classes等）

【重要】
- Streamlit Cloud 上不要跑训练；请在本机运行本脚本生成 X_test.csv + XGB.pkl，
  然后把这两个文件一起提交到 GitHub 仓库根目录，Cloud 只做推理。
===========================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier

from skopt import BayesSearchCV
from skopt.space import Real, Integer

import shap
from PIL import Image
import joblib

warnings.filterwarnings("ignore")

# =========================
# 0) 画图字体
# =========================
plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 1) 路径 & 输出目录（你按需修改）
# =========================
excel_path_win = r"C:\Users\weien\Desktop\sic\sic血小板\sic血小板数据\用于XGBOOST的4组数据.xlsx"

# 输出（建议：就是你 GitHub 仓库根目录）
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

result_dir = os.path.join(OUT_DIR, "results_dev_4class_shap_bee_left_bar_right")
os.makedirs(result_dir, exist_ok=True)

data_path = excel_path_win
if not os.path.exists(data_path):
    raise FileNotFoundError(f"找不到Excel文件：{excel_path_win}")

print(f"Using data path: {data_path}")
print(f"Results will be saved to: {os.path.abspath(result_dir)}")
print(f"Artifacts will be saved to: {OUT_DIR}")

# =========================
# 2) 读取 Excel + 清理列名
# =========================
df = pd.read_excel(data_path)
df.columns = df.columns.astype(str).str.strip()

# =========================
# 3) 标签列 & 特征列（最终24个）
# =========================
FEATURE_COLS_FINAL24 = [
    "age", "aki", "lc", "hf", "sapsii",
    "hematocrit", "hemoglobin", "platelet",
    "rdw", "rbc", "wbc",
    "anion_gap", "chloride", "glucose", "sodium", "lac",
    "creatinine", "bun",
    "hr", "rr", "temperature",
    "inr", "pt", "aptt"
]

# 大小写不敏感匹配
col_map = {c.lower(): c for c in df.columns}
def pick_col(name: str):
    return col_map.get(str(name).lower(), None)

# 目标列（Group/group）
target_real = pick_col("group")
if target_real is None:
    raise ValueError("找不到目标列 'Group/group'（大小写不敏感也没找到）。")

# 特征列：严格模式（缺一个就报错）
feature_real_cols = []
missing_cols = []
for c in FEATURE_COLS_FINAL24:
    real = pick_col(c)
    if real is None:
        missing_cols.append(c)
    else:
        feature_real_cols.append(real)

if missing_cols:
    raise ValueError(
        "以下特征列在Excel中没找到（请检查列名拼写/大小写/空格）：\n"
        + ", ".join(missing_cols)
    )

# 保险：剔除ID/泄漏列（即便误放进来）
for forbidden in ["subject_id", "hadm_id", "stay_id", "death_within_icu_28days"]:
    fcol = pick_col(forbidden)
    if fcol is not None and fcol in feature_real_cols:
        feature_real_cols.remove(fcol)

print("\n[Info] Final feature columns used:")
print(feature_real_cols)
print(f"[Info] n_features = {len(feature_real_cols)}")

# =========================
# 4) 构造 X, y
# =========================
y_raw = df[target_real]
X = df[feature_real_cols].copy()

le = LabelEncoder()
y = le.fit_transform(y_raw)

print("\nClass mapping (original -> encoded):")
for k, v in dict(zip(le.classes_, le.transform(le.classes_))).items():
    print(f"  {k} -> {v}")

n_classes = len(le.classes_)
print(f"\n[Info] Detected n_classes = {n_classes}")
if n_classes != 4:
    print("[Warning] 你说是4组，但检测到类别数不是4；请确认 group 是否有缺失/异常。")

print("\nOriginal class distribution:")
print(pd.Series(y).value_counts().sort_index())

# =========================
# 5) 数值化 + 处理inf
# =========================
X = X.replace([np.inf, -np.inf], np.nan)
X = X.apply(pd.to_numeric, errors="coerce")

# =========================
# 6) 划分训练/测试 + 用训练集中位数填补缺失（避免泄漏）
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=666, stratify=y
)

train_median = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_median).fillna(0).astype(np.float32)
X_test = X_test.fillna(train_median).fillna(0).astype(np.float32)

print("\nTrain class distribution:")
print(pd.Series(y_train).value_counts().sort_index())
print("\nTest class distribution:")
print(pd.Series(y_test).value_counts().sort_index())

# 额外：保存 X_test.csv（给 Streamlit LIME 用）
X_test_csv_path = os.path.join(OUT_DIR, "X_test.csv")
pd.DataFrame(X_test, columns=feature_real_cols).to_csv(X_test_csv_path, index=False, encoding="utf-8-sig")
print("\n[OK] Saved X_test.csv ->", X_test_csv_path)

# =========================
# 7) 评估函数（报告+混淆矩阵+ROC）
# =========================
def _get_classes_order(model, fallback_y):
    if hasattr(model, "classes_"):
        return np.array(model.classes_)
    if hasattr(model, "named_steps") and "xgb" in model.named_steps and hasattr(model.named_steps["xgb"], "classes_"):
        return np.array(model.named_steps["xgb"].classes_)
    return np.unique(fallback_y)

def evaluate_model(model, X_eval, y_true, dataset_name="Dataset", save_suffix=""):
    y_pred = model.predict(X_eval)
    y_pred_proba = model.predict_proba(X_eval)

    target_names = [str(c) for c in le.classes_]
    labels = list(range(len(le.classes_)))

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        digits=4
    )
    print(f"\n{dataset_name} Classification Report ({save_suffix}):\n{report}")

    with open(os.path.join(result_dir, f"classification_report_{save_suffix}.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=target_names,
        cmap="Blues",
        normalize=None,
        ax=plt.gca()
    )
    plt.title(f"{dataset_name} Confusion Matrix ({save_suffix})")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"confusion_matrix_{save_suffix}.png"), dpi=300)
    plt.close()

    # ROC (OvR)
    classes_order = _get_classes_order(model, y_true)

    plt.figure(figsize=(10, 8))
    for idx, c in enumerate(classes_order):
        y_binary = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Group {le.inverse_transform([c])[0]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"{dataset_name} ROC Curve ({save_suffix})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"roc_curve_{save_suffix}.png"), dpi=300)
    plt.close()

# =========================
# 8) 初始模型（不调参）
# =========================
initial_model = XGBClassifier(
    objective="multi:softprob",
    num_class=n_classes,
    random_state=666,
    eval_metric="mlogloss",
    verbosity=0,
    tree_method="hist"
)
initial_model.fit(X_train, y_train)
evaluate_model(initial_model, X_test, y_test, dataset_name="Dev(Test)", save_suffix="initial")

# =========================
# 9) SMOTE + BayesSearchCV（调参）
# =========================
counts = pd.Series(y_train).value_counts()
min_count = int(counts.min())

use_smote = True
if min_count <= 1:
    use_smote = False
    print("\n[Warning] 最小类别样本数<=1，SMOTE无法使用，将跳过SMOTE。")

k_neighbors = 5
if use_smote:
    k_neighbors = min(5, min_count - 1)
    if k_neighbors < 1:
        use_smote = False
        print("\n[Warning] 样本太少无法设置有效 k_neighbors，跳过SMOTE。")

if use_smote:
    pipe = Pipeline(steps=[
        ("smote", SMOTE(random_state=666, k_neighbors=k_neighbors)),
        ("xgb", XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            random_state=666,
            eval_metric="mlogloss",
            verbosity=0,
            tree_method="hist"
        ))
    ])
    print(f"\n[Info] SMOTE enabled, k_neighbors={k_neighbors}")
else:
    pipe = Pipeline(steps=[
        ("xgb", XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            random_state=666,
            eval_metric="mlogloss",
            verbosity=0,
            tree_method="hist"
        ))
    ])
    print("\n[Info] SMOTE disabled.")

param_space = {
    "xgb__max_depth": Integer(3, 10),
    "xgb__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
    "xgb__n_estimators": Integer(50, 500),
    "xgb__subsample": Real(0.6, 1.0),
    "xgb__colsample_bytree": Real(0.6, 1.0),
    "xgb__gamma": Real(0, 10),
    "xgb__min_child_weight": Integer(1, 10),
    "xgb__reg_alpha": Real(0, 5),
    "xgb__reg_lambda": Real(0, 5),
}

bayes_search = BayesSearchCV(
    estimator=pipe,
    search_spaces=param_space,
    cv=5,
    n_iter=50,
    scoring="f1_macro",
    random_state=666,
    verbose=1,
    n_jobs=-1
)
bayes_search.fit(X_train, y_train)

print("\nBest params:")
print(bayes_search.best_params_)

best_model = bayes_search.best_estimator_
evaluate_model(best_model, X_test, y_test, dataset_name="Dev(Test)", save_suffix="best")

# =========================
# 10) SHAP：蜂群左 + 贡献条形右（每类一张）
# =========================
def _matrix_for_class(shap_values, class_idx: int):
    vals = shap_values.values if hasattr(shap_values, "values") else shap_values
    if isinstance(vals, list):
        return vals[class_idx]
    vals = np.asarray(vals)
    if vals.ndim == 3:
        return vals[:, :, class_idx]
    if vals.ndim == 2:
        return vals
    raise ValueError(f"Unsupported SHAP values shape: {getattr(vals, 'shape', None)}")

def save_shap_beeswarm_left_bar_right(
    shap_values,
    X_df: pd.DataFrame,
    class_idx: int,
    out_path: str,
    title_left: str,
    title_right: str,
    dpi: int = 300,
    max_display: int = 20,
):
    base, _ = os.path.splitext(out_path)
    tmp_bee = base + f"_tmp_bee_c{class_idx}.png"
    tmp_bar = base + f"_tmp_bar_c{class_idx}.png"

    mat = _matrix_for_class(shap_values, class_idx)

    # beeswarm (LEFT)
    plt.figure()
    shap.summary_plot(
        mat,
        X_df,
        max_display=max_display,
        show=False
    )
    plt.title(title_left)
    plt.tight_layout()
    plt.savefig(tmp_bee, dpi=dpi, bbox_inches="tight")
    plt.close()

    # bar (RIGHT)
    plt.figure()
    shap.summary_plot(
        mat,
        X_df,
        plot_type="bar",
        max_display=max_display,
        show=False
    )
    plt.title(title_right)
    plt.tight_layout()
    plt.savefig(tmp_bar, dpi=dpi, bbox_inches="tight")
    plt.close()

    # stitch horizontally
    img_left = Image.open(tmp_bee).convert("RGB")
    img_right = Image.open(tmp_bar).convert("RGB")

    new_w = img_left.width + img_right.width
    new_h = max(img_left.height, img_right.height)
    canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    canvas.paste(img_left, (0, (new_h - img_left.height) // 2))
    canvas.paste(img_right, (img_left.width, (new_h - img_right.height) // 2))

    out_path = base + ".png"
    canvas.save(out_path, quality=95)

    # cleanup
    for p in [tmp_bee, tmp_bar]:
        try:
            os.remove(p)
        except OSError:
            pass

X_test_df = pd.DataFrame(X_test, columns=feature_real_cols)

xgb_best = best_model.named_steps["xgb"]
explainer = shap.TreeExplainer(xgb_best)
shap_values = explainer(X_test_df)

for class_idx in range(n_classes):
    group_name = str(le.inverse_transform([class_idx])[0])
    out_file = os.path.join(result_dir, f"shap_combined_group_{group_name}_bee_left_bar_right.png")
    save_shap_beeswarm_left_bar_right(
        shap_values=shap_values,
        X_df=X_test_df,
        class_idx=class_idx,
        out_path=out_file,
        title_left=f"Group {group_name} | Beeswarm",
        title_right=f"Group {group_name} | Bar",
        dpi=300,
        max_display=min(24, len(feature_real_cols))
    )

# =========================
# 11) 导出模型给 Streamlit 用（XGB.pkl）
# =========================
bundle = {
    "model": best_model,                 # pipeline (可能含smote) + xgb
    "feature_cols": feature_real_cols,   # 24个特征列（Excel真实列名）
    "train_median": train_median,        # 缺失填补用
    "classes": list(le.classes_),        # 4组原始名字
    "n_classes": int(n_classes),
    "best_params": getattr(bayes_search, "best_params_", None),
}

model_path = os.path.join(OUT_DIR, "XGB.pkl")
joblib.dump(bundle, model_path, compress=3)
print("\n[OK] Saved XGB.pkl ->", model_path)

print("\nAll done ✅")
print("Results:", os.path.abspath(result_dir))
