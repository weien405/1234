import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="SIC血小板 四分类 Group 预测（XGBoost）", layout="wide")

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "XGB.pkl"   # 训练脚本导出
XTEST_PATH = APP_DIR / "X_test.csv"

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_x_test():
    return pd.read_csv(XTEST_PATH)

st.title("SIC血小板 四分类 Group 预测（XGBoost）")
st.caption("Cloud 只做推理：请先在本机运行 train_export_xgb_4class.py 生成 XGB.pkl 和 X_test.csv，再推到 GitHub。")

# ---- safe load ----
try:
    bundle = load_bundle()
except Exception as e:
    st.error("无法加载 XGB.pkl。请确认文件已提交到仓库根目录，并且 requirements.txt 包含 xgboost / imbalanced-learn 等依赖。")
    st.exception(e)
    st.stop()

try:
    X_test = load_x_test()
except Exception as e:
    st.warning("无法加载 X_test.csv（LIME 会不可用）。请确认 X_test.csv 在仓库根目录。")
    X_test = None

model = bundle["model"]
feature_cols = list(bundle["feature_cols"])
classes = [str(c) for c in bundle["classes"]]
n_classes = int(bundle.get("n_classes", len(classes)))
train_median = pd.Series(bundle.get("train_median", pd.Series(dtype=float)))

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    if not train_median.empty:
        df = df.fillna(train_median)
    df = df.fillna(0).astype(np.float32)
    return df

# -------- input UI (24 features) --------
st.subheader("输入特征（24项）")
with st.form("input_form"):
    cols = st.columns(3)
    inputs = {}
    for i, feat in enumerate(feature_cols):
        with cols[i % 3]:
            dv = float(train_median.get(feat, 0.0)) if not train_median.empty else 0.0
            inputs[feat] = st.number_input(feat, value=dv, step=0.1, format="%.4f")
    submitted = st.form_submit_button("Predict")

if submitted:
    x_row = pd.DataFrame([inputs], columns=feature_cols)
    x_row = preprocess(x_row)

    # predict
    try:
        proba = model.predict_proba(x_row)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
    except Exception as e:
        st.error("预测失败：请确认模型与特征列顺序一致（24列），以及模型文件有效。")
        st.exception(e)
        st.stop()

    st.subheader("预测结果")
    c1, c2 = st.columns([1, 1])
    c1.metric("预测组（group）", pred_label)
    c2.metric("最高概率", f"{float(proba[pred_idx]):.4f}")

    df_prob = pd.DataFrame({"Group": classes, "Probability": [float(p) for p in proba]}).sort_values("Probability", ascending=False)
    st.dataframe(df_prob, use_container_width=True)

    # -------- SHAP (single instance) --------
    st.subheader("SHAP 单样本解释（预测组）")
    try:
        import shap  # lazy import

        # 从pipeline中取出 xgb
        xgb = None
        if hasattr(model, "named_steps") and "xgb" in model.named_steps:
            xgb = model.named_steps["xgb"]
        else:
            xgb = model

        explainer = shap.TreeExplainer(xgb)
        shap_exp = explainer(x_row)  # shap.Explanation

        vals = shap_exp.values if hasattr(shap_exp, "values") else shap_exp
        vals = np.asarray(vals)

        # 多分类常见形状：(1, n_features, n_classes)
        if vals.ndim == 3:
            contrib = vals[0, :, pred_idx]
        elif vals.ndim == 2:
            contrib = vals[0, :]
        else:
            contrib = vals.reshape(-1)

        feat_names = np.array(feature_cols)
        top_n = min(15, len(contrib))
        top_idx = np.argsort(np.abs(contrib))[-top_n:]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(feat_names[top_idx], contrib[top_idx])
        ax.set_title(f"Top {top_n} SHAP Contributions | Pred={pred_label}")
        ax.set_xlabel("SHAP value (impact on model output)")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning("SHAP 生成失败（多为环境/版本差异）。")
        st.exception(e)

    # -------- LIME (multi-class) --------
    st.subheader("LIME 单样本解释")
    if X_test is None:
        st.info("没有 X_test.csv，LIME 不可用。")
    else:
        try:
            # 背景数据必须和 feature_cols 对齐
            X_bg = X_test[feature_cols].copy()
            X_bg = preprocess(X_bg)

            explainer = LimeTabularExplainer(
                training_data=X_bg.values,
                feature_names=feature_cols,
                class_names=classes,
                mode="classification"
            )

            def predict_fn(x_np):
                x_df = pd.DataFrame(x_np, columns=feature_cols)
                x_df = preprocess(x_df)
                return model.predict_proba(x_df)

            exp = explainer.explain_instance(
                data_row=x_row.values[0],
                predict_fn=predict_fn,
                num_features=min(15, len(feature_cols))
            )
            st.components.v1.html(exp.as_html(show_table=False), height=800, scrolling=True)
        except Exception as e:
            st.warning("LIME 生成失败。")
            st.exception(e)

# ---- optional: show training plots if you pushed results dir ----
with st.expander("查看训练阶段生成的评估图/SHAP合并图（如果你已把 results 目录推到GitHub）", expanded=False):
    results_dir = APP_DIR / "results_dev_4class_shap_bee_left_bar_right"
    if results_dir.exists():
        imgs = sorted(list(results_dir.glob("*.png")))
        if imgs:
            for p in imgs:
                st.image(str(p), caption=p.name, use_container_width=True)
        else:
            st.info("results 目录存在，但没有 png 文件。")
    else:
        st.info("未检测到 results_dev_4class_shap_bee_left_bar_right 目录。")
