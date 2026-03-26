# -*- coding: utf-8 -*-
"""
analysis.py — Olist E-Commerce Analysis Pipeline

Sections:
    1. EDA  (3 figures)
    2. Geographic & Corridor Analysis — RQ1  (3 figures)
    3. Satisfaction Prediction — RQ2  (3 figures)
    4. Seller Benchmarking — RQ3  (2 figures)

Prerequisites: Run data_preparation.py first.

Usage:
    python analysis.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report, f1_score,
    precision_score, recall_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(SCRIPT_DIR, "outputs", "train_data.csv")
TEST_PATH   = os.path.join(SCRIPT_DIR, "outputs", "test_data.csv")
MASTER_PATH = os.path.join(SCRIPT_DIR, "outputs", "master_olist_data.csv")
FIG_DIR     = os.path.join(SCRIPT_DIR, "outputs", "figures")
RES_DIR     = os.path.join(SCRIPT_DIR, "outputs", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

SEED = 42

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "serif", "font.size": 11,
})
sns.set_style("whitegrid")
PALETTE = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5"]


# ── Seller Features (leakage-safe) ──────────────────────────────────

def add_seller_features(train_df, test_df):
    """
    Aggregate seller-level features from training data only, then map
    to both sets.  Unseen test-period sellers receive training medians.
    """
    print("   Computing seller features from training data only")
    seller_stats = train_df.groupby("seller_id").agg(
        seller_order_count=("order_id", "count"),
        seller_avg_review=("review_score", "mean"),
        seller_late_rate=("is_late", "mean"),
        seller_avg_distance=("distance_km", "mean"),
    ).reset_index()

    train_out = train_df.merge(seller_stats, on="seller_id", how="left")
    test_out  = test_df.merge(seller_stats, on="seller_id", how="left")

    for col in seller_stats.columns.drop("seller_id"):
        med = train_out[col].median()
        train_out[col] = train_out[col].fillna(med)
        test_out[col]  = test_out[col].fillna(med)  # Unseen test sellers get train medians (realistic cold-start)

    n_seen = test_df["seller_id"].isin(seller_stats["seller_id"]).sum()
    print(f"   Sellers in training: {len(seller_stats):,}")
    print(f"   Test sellers seen: {n_seen:,} / {len(test_df):,}")
    return train_out, test_out


# ── Section 1: EDA ──────────────────────────────────────────────────

def run_eda(df):
    """Figures 1–3: review distribution, satisfaction by delay, correlations."""
    print("\n1  EDA")

    # Figure 1 — review scores & class balance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    scores = df["review_score"].value_counts().sort_index()
    colors = ["#c0392b", "#e67e22", "#f1c40f", "#27ae60", "#1a9850"]
    ax1.bar(scores.index, scores.values, color=colors, edgecolor="white")
    for idx, val in scores.items():
        ax1.text(idx, val + 500, f"{val:,}", ha="center", fontsize=9)
    ax1.set_xlabel("Review Score")
    ax1.set_ylabel("Count")
    ax1.set_title("(a) Review Score Distribution")

    cls = df["satisfied"].value_counts()
    ax2.pie(cls.values, labels=["Satisfied (4-5)", "Dissatisfied (1-3)"],
            colors=["#27ae60", "#c0392b"], autopct="%1.1f%%",
            startangle=90, explode=[0, 0.05])
    ax2.set_title("(b) Target Class Balance")
    fig.suptitle("Figure 1: Review Score & Class Distribution", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_01_review_distribution.png"),
                bbox_inches="tight")
    plt.close()
    print("   fig_01")

    # Figure 2 — satisfaction by delivery delay band
    df = df.copy()
    df["delay_band"] = pd.cut(
        df["delivery_delay_days"],
        bins=[-np.inf, -7, 0, 7, np.inf],
        labels=["Very Early\n(>7d early)", "On Time\n(0-7d early)",
                "Slight Delay\n(0-7d late)", "Severe Delay\n(>7d late)"],
    )
    delay_sat = df.groupby("delay_band", observed=True).agg(
        sat_rate=("satisfied", "mean"), count=("order_id", "count"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_c = ["#27ae60", "#f1c40f", "#e67e22", "#c0392b"]
    bars = ax.bar(delay_sat["delay_band"], delay_sat["sat_rate"] * 100,
                  color=bar_c, edgecolor="black", linewidth=0.4)
    ax.axhline(y=df["satisfied"].mean() * 100, color="black", linestyle="--",
               label=f'Overall: {df["satisfied"].mean()*100:.1f}%')
    for bar, (_, row) in zip(bars, delay_sat.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'n={row["count"]:,}', ha="center", fontsize=9)
    ax.set_ylabel("Satisfaction Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Figure 2: Satisfaction Rate by Delivery Performance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_02_satisfaction_by_delay.png"),
                bbox_inches="tight")
    plt.close()
    print("   fig_02")

    # Figure 3 — correlation heatmap
    corr_cols = [
        "distance_km", "actual_delivery_days", "delivery_delay_days",
        "total_order_value", "freight_ratio", "num_items",
        "payment_installments", "satisfied",
    ]
    corr = df[corr_cols].dropna().corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax)
    ax.set_title("Figure 3: Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_03_correlation_heatmap.png"),
                bbox_inches="tight")
    plt.close()
    print("   fig_03")


# ── Section 2: Geographic & Corridor Analysis (RQ1) ─────────────────

def run_corridor_analysis(df):
    """
    RQ1: Effect of geographic distance on satisfaction.
    Figures 4–6; Mann–Whitney U, Spearman corridor correlation.
    """
    print("\n2  Geographic & Corridor Analysis (RQ1)")

    # Mann–Whitney U: same vs cross-state
    same  = df[df["same_state"] == 1]["review_score"]
    cross = df[df["same_state"] == 0]["review_score"]
    U, p_mw = stats.mannwhitneyu(same, cross, alternative="greater")
    r_rb = 1 - (2 * U) / (len(same) * len(cross))

    same_sat  = df[df["same_state"] == 1]["satisfied"].mean()
    cross_sat = df[df["same_state"] == 0]["satisfied"].mean()
    print(f"   Same-state:  {100*same_sat:.1f}% (n={len(same):,})")
    print(f"   Cross-state: {100*cross_sat:.1f}% (n={len(cross):,})")
    print(f"   U={U:,.0f}, p={p_mw:.2e}, r={r_rb:.4f}")

    # Figure 4 — same vs cross-state
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    grp = df.groupby("same_state").agg(
        sat_rate=("satisfied", "mean"), count=("order_id", "count"),
        avg_delay=("delivery_delay_days", "mean"),
    ).reset_index()
    grp["label"] = grp["same_state"].map({1: "Same State", 0: "Cross State"})

    ax1.bar(grp["label"], grp["sat_rate"] * 100,
            color=[PALETTE[0], PALETTE[1]], edgecolor="black", linewidth=0.4)
    for i, (_, r) in enumerate(grp.iterrows()):
        ax1.text(i, r["sat_rate"] * 100 + 1,
                 f'{r["sat_rate"]*100:.1f}%\nn={r["count"]:,}',
                 ha="center", fontsize=10)
    ax1.set_ylabel("Satisfaction Rate (%)")
    ax1.set_title(f"(a) Satisfaction Rate\n(p={p_mw:.2e}, r={r_rb:.3f})")
    ax1.set_ylim(0, 100)

    ax2.bar(grp["label"], grp["avg_delay"],
            color=[PALETTE[0], PALETTE[1]], edgecolor="black", linewidth=0.4)
    for i, (_, r) in enumerate(grp.iterrows()):
        ax2.text(i, r["avg_delay"] - 0.4, f'{r["avg_delay"]:.1f}d',
                 ha="center", fontweight="bold", color="white")
    ax2.set_ylabel("Avg Delivery Delay (days)")
    ax2.set_title("(b) Average Delivery Delay")
    fig.suptitle("Figure 4: Same-State vs Cross-State Delivery", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_04_same_vs_cross_state.png"),
                bbox_inches="tight")
    plt.close()
    print("   fig_04")

    # Corridor-level aggregation (min 50 orders)
    corr_df = df.groupby("corridor").agg(
        n=("order_id", "count"), sat_rate=("satisfied", "mean"),
        avg_distance=("distance_km", "mean"), late_rate=("is_late", "mean"),
    ).reset_index()
    corr_df = corr_df[corr_df.n >= 50]

    rho, p_sp = stats.spearmanr(corr_df["avg_distance"], corr_df["sat_rate"])
    print(f"   Spearman ρ={rho:.3f}, p={p_sp:.4f}")

    # Figure 5 — distance vs satisfaction scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        corr_df["avg_distance"], corr_df["sat_rate"] * 100,
        s=corr_df["n"] / 5, alpha=0.6, c=corr_df["late_rate"],
        cmap="RdYlGn_r", edgecolors="black", linewidth=0.3,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Late Delivery Rate")
    z = np.polyfit(corr_df["avg_distance"], corr_df["sat_rate"] * 100, 1)
    x_line = np.linspace(corr_df["avg_distance"].min(),
                         corr_df["avg_distance"].max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="red", linewidth=1.5,
            label=f"Trend (ρ={rho:.3f}, p={p_sp:.4f})")
    ax.set_xlabel("Average Shipping Distance (km)")
    ax.set_ylabel("Satisfaction Rate (%)")
    ax.set_title("Figure 5: Distance vs Satisfaction by Corridor\n"
                 "(bubble size = volume, colour = late rate)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_05_distance_vs_satisfaction.png"),
                bbox_inches="tight")
    plt.close()
    print("   fig_05")

    # Figure 6 — satisfaction by customer state
    state_sat = df.groupby("customer_state").agg(
        n=("order_id", "count"), sat_rate=("satisfied", "mean"),
    ).reset_index().sort_values("sat_rate")

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = [PALETTE[0] if r > df["satisfied"].mean() else "#c0392b"
              for r in state_sat["sat_rate"]]
    ax.barh(state_sat["customer_state"], state_sat["sat_rate"] * 100,
            color=colors, edgecolor="white")
    ax.axvline(x=df["satisfied"].mean() * 100, color="black", linestyle="--",
               label=f'National avg: {df["satisfied"].mean()*100:.1f}%')
    for i, (_, r) in enumerate(state_sat.iterrows()):
        ax.text(r["sat_rate"] * 100 + 0.3, i,
                f'{r["sat_rate"]*100:.1f}%', va="center", fontsize=8)
    ax.set_xlabel("Satisfaction Rate (%)")
    ax.set_title("Figure 6: Customer Satisfaction by State")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_06_satisfaction_by_state.png"),
                bbox_inches="tight")
    plt.close()
    print("   fig_06")

    # Export results
    corr_df.to_csv(os.path.join(RES_DIR, "corridor_analysis.csv"), index=False)
    stat_res = pd.DataFrame([
        {"Test": "Mann-Whitney U (same vs cross-state)", "Statistic": U,
         "p_value": p_mw, "Effect_Size": f"rank-biserial r={r_rb:.4f}"},
        {"Test": "Spearman (corridor distance vs satisfaction)",
         "Statistic": rho, "p_value": p_sp, "Effect_Size": f"rho={rho:.3f}"},
    ])
    stat_res.to_csv(os.path.join(RES_DIR, "statistical_tests.csv"), index=False)
    print("   corridor_analysis.csv, statistical_tests.csv")


# ── Section 3: Satisfaction Prediction (RQ2) ─────────────────────────

def run_prediction(train_df, test_df):
    """
    RQ2: Predict customer satisfaction.
    Models: Logistic Regression (baseline), XGBoost (tuned).
    SMOTE applied to training data only.
    Figures 7–9; model comparison and feature importance CSVs.
    """
    print("\n3  Satisfaction Prediction (RQ2)")

    feature_cols = [
        "distance_km", "actual_delivery_days", "delivery_delay_days",
        "carrier_handling_days",
        "total_order_value", "freight_ratio", "num_items",
        "payment_installments", "pay_credit_card", "pay_boleto", "pay_debit_card",
        "product_weight_g", "same_state",
        "purchase_month", "purchase_dow", "purchase_hour",
        "seller_order_count", "seller_avg_review", "seller_late_rate",
        "seller_avg_distance",
    ]
    cat_cols = [c for c in train_df.columns if c.startswith("cat_")]  # Dummies for top-5 categories (80% coverage, dimensionality control)
    feature_cols.extend(cat_cols)

    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()
    y_train = train_df["satisfied"].copy()
    y_test  = test_df["satisfied"].copy()

    # Impute NaNs with training medians
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test  = X_test.fillna(medians)

    print(f"   Train: {len(X_train):,}  Test: {len(X_test):,}  "
          f"Features: {len(feature_cols)}")

    # SMOTE on training data only
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=SEED)
        X_tr, y_tr = sm.fit_resample(X_train, y_train)
        print(f"   SMOTE → {(y_tr==0).sum():,} dissat / {(y_tr==1).sum():,} sat")
    except ImportError:
        print("   WARNING: imblearn not installed — skipping SMOTE")
        X_tr, y_tr = X_train, y_train

    # Logistic Regression
    print("   Training Logistic Regression …")
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1_000, random_state=SEED, class_weight="balanced")),
    ])
    lr.fit(X_tr, y_tr)

    # XGBoost
    has_xgb = False
    try:
        from xgboost import XGBClassifier
        print("   Training XGBoost (RandomizedSearchCV, 20 iter, 5-fold) …")
        param_dist = {
            "n_estimators": [100, 150, 200],
            "max_depth": [4, 5, 6, 7],
            "learning_rate": [0.05, 0.1, 0.15],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
            "min_child_weight": [1, 3, 5],
        }
        xgb_base = XGBClassifier(random_state=SEED, eval_metric="logloss")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        search = RandomizedSearchCV(
            xgb_base, param_dist, n_iter=20, cv=cv,
            scoring="f1", random_state=SEED, n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        xgb = search.best_estimator_
        has_xgb = True
        print(f"   Best CV F1: {search.best_score_:.4f}")
        print(f"   Best params: {search.best_params_}")
    except ImportError:
        print("   WARNING: xgboost not installed")

    # Evaluate
    results = {}
    models = [("Logistic Regression", lr)]
    if has_xgb:
        models.append(("XGBoost", xgb))

    for name, model in models:
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "pred": pred, "prob": prob,
            "acc":    accuracy_score(y_test, pred),
            "auc":    roc_auc_score(y_test, prob),
            "f1_d":   f1_score(y_test, pred, pos_label=0),
            "prec_d": precision_score(y_test, pred, pos_label=0),
            "rec_d":  recall_score(y_test, pred, pos_label=0),
        }
        r = results[name]
        print(f"\n   {name}:")
        print(f"     Acc={r['acc']:.4f}  AUC={r['auc']:.4f}  "
              f"F1(dissat)={r['f1_d']:.4f}")
        print(classification_report(y_test, pred,
              target_names=["Dissatisfied", "Satisfied"]))

    # Export model comparison
    comp = pd.DataFrame([{
        "Model": n, "Accuracy": f"{r['acc']:.4f}", "ROC_AUC": f"{r['auc']:.4f}",
        "Precision_Dissat": f"{r['prec_d']:.4f}",
        "Recall_Dissat": f"{r['rec_d']:.4f}",
        "F1_Dissat": f"{r['f1_d']:.4f}",
    } for n, r in results.items()])
    comp.to_csv(os.path.join(RES_DIR, "model_comparison.csv"), index=False)

    # Figure 7 — confusion matrices
    n_mod = len(results)
    fig, axes = plt.subplots(1, n_mod, figsize=(6 * n_mod, 5))
    if n_mod == 1:
        axes = [axes]
    for i, (name, r) in enumerate(results.items()):
        cm = confusion_matrix(y_test, r["pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Dissat.", "Sat."],
                    yticklabels=["Dissat.", "Sat."])
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        axes[i].set_title(f"{name}\nAcc={r['acc']:.3f}, AUC={r['auc']:.3f}")
    fig.suptitle("Figure 7: Confusion Matrices", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_07_confusion_matrices.png"),
                bbox_inches="tight")
    plt.close()
    print("   fig_07")

    # Figure 8 — ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, r), c in zip(results.items(), PALETTE):
        fpr, tpr, _ = roc_curve(y_test, r["prob"])
        ax.plot(fpr, tpr, linewidth=2, color=c,
                label=f'{name} (AUC={r["auc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Figure 8: ROC Curve Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_08_roc_curves.png"),
                bbox_inches="tight")
    plt.close()
    print("   fig_08")

    # Figure 9 — feature importance (gain + permutation)
    if has_xgb:
        gain_imp = pd.DataFrame({
            "Feature": feature_cols,
            "Gain": xgb.feature_importances_,
        })
        print("   Computing permutation importance (10 repeats) …")
        perm = permutation_importance(
            xgb, X_test, y_test, n_repeats=10,
            random_state=SEED, scoring="roc_auc",
        )
        gain_imp["Permutation"] = perm.importances_mean
        gain_imp = gain_imp.sort_values("Permutation", ascending=True).tail(15)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
        ax1.barh(gain_imp["Feature"], gain_imp["Gain"],
                 color=PALETTE[0], edgecolor="white")
        ax1.set_xlabel("Importance (Gain)")
        ax1.set_title("(a) Gain-Based Importance")

        ax2.barh(gain_imp["Feature"], gain_imp["Permutation"],
                 color=PALETTE[1], edgecolor="white")
        ax2.set_xlabel("Importance (Permutation, AUC drop)")
        ax2.set_title("(b) Permutation Importance")

        fig.suptitle("Figure 9: XGBoost Feature Importance (Top 15)", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig_09_feature_importance.png"),
                    bbox_inches="tight")
        plt.close()
        print("   fig_09")

        # Full ranking export
        full = pd.DataFrame({
            "Feature": feature_cols,
            "Gain_Importance": xgb.feature_importances_,
            "Permutation_Importance": perm.importances_mean,
        }).sort_values("Permutation_Importance", ascending=False)
        full.to_csv(os.path.join(RES_DIR, "feature_importance.csv"), index=False)

    print("   model_comparison.csv, feature_importance.csv")
    return results


# ── Section 4: Seller Performance Benchmarking (RQ3) ─────────────────

def run_seller_analysis(df):
    """
    RQ3: K-Means seller segmentation.  Min k=3 enforced.
    Figures 10–11; seller_segments.csv, seller_recommendations.csv.
    """
    print("\n4  Seller Benchmarking (RQ3)")

    seller = df.groupby("seller_id").agg(
        order_count=("order_id", "count"),
        avg_review=("review_score", "mean"),
        sat_rate=("satisfied", "mean"),
        late_rate=("is_late", "mean"),
        avg_distance=("distance_km", "mean"),
        avg_delivery_days=("actual_delivery_days", "mean"),
        avg_order_value=("total_order_value", "mean"),
    ).reset_index()

    seller = seller[seller.order_count >= 5].copy()
    print(f"   Sellers with ≥5 orders: {len(seller):,}")

    # Clustering
    clust_cols = ["avg_review", "late_rate", "avg_distance",
                  "avg_delivery_days", "order_count", "avg_order_value"]
    X_raw = seller[clust_cols].copy().fillna(seller[clust_cols].median())
    for col in ["order_count", "avg_order_value", "avg_distance"]:
        X_raw[col] = np.log1p(X_raw[col])

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw)

    # Silhouette for k=2..6
    k_range = range(2, 7)
    sils = []
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X_sc)
        s = silhouette_score(X_sc, labels)
        sils.append(s)
        print(f"     k={k}: silhouette={s:.4f}")

    best_k = max(3, list(k_range)[np.argmax(sils)])
    print(f"   Selected k={best_k}")

    km = KMeans(n_clusters=best_k, n_init=20, random_state=SEED)
    seller["segment"] = km.fit_predict(X_sc)

    # Name segments by characteristics
    seg_means = seller.groupby("segment")[clust_cols + ["sat_rate"]].mean()
    names = {}
    for seg in range(best_k):
        s = seg_means.loc[seg]
        if (s["avg_review"] >= seg_means["avg_review"].quantile(0.7)
                and s["late_rate"] <= seg_means["late_rate"].quantile(0.3)):
            names[seg] = "Top Performers"
        elif s["late_rate"] >= seg_means["late_rate"].quantile(0.7):
            names[seg] = "At-Risk"
        elif s["order_count"] >= seg_means["order_count"].median():
            names[seg] = "High Volume"
        else:
            names[seg] = "Developing"

    # Resolve duplicate names
    used = {}
    for seg in names:
        base = names[seg]
        if base in used:
            used[base] += 1
            names[seg] = f"{base} ({used[base]})"
        else:
            used[base] = 1

    seller["segment_name"] = seller["segment"].map(names)

    # Profile
    profile = seller.groupby("segment_name").agg(
        n=("seller_id", "count"), avg_review=("avg_review", "mean"),
        sat_rate=("sat_rate", "mean"), late_rate=("late_rate", "mean"),
        avg_orders=("order_count", "mean"), avg_distance=("avg_distance", "mean"),
    ).round(3)
    print(f"\n{profile.to_string()}")
    profile.to_csv(os.path.join(RES_DIR, "seller_segments.csv"))

    # Figure 10 — radar chart
    radar_cols = ["avg_review", "sat_rate", "late_rate", "avg_distance",
                  "avg_delivery_days", "order_count"]
    seg_radar = seller.groupby("segment_name")[radar_cols].mean()
    seg_norm = (seg_radar - seg_radar.min()) / (seg_radar.max() - seg_radar.min())
    for col in ["late_rate", "avg_distance", "avg_delivery_days"]:
        seg_norm[col] = 1 - seg_norm[col]

    angles = np.linspace(0, 2 * np.pi, len(radar_cols), endpoint=False).tolist()
    angles += angles[:1]
    labels_r = ["Avg Review", "Satisfaction", "Low Late Rate",
                "Short Distance", "Fast Delivery", "Order Volume"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (name, vals) in enumerate(seg_norm.iterrows()):
        v = vals.tolist() + [vals.tolist()[0]]
        ax.plot(angles, v, "o-", linewidth=1.5, label=name,
                color=PALETTE[i % len(PALETTE)])
        ax.fill(angles, v, alpha=0.08, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_r, fontsize=9)
    ax.set_title("Figure 10: Seller Segment Profiles", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_10_seller_segments.png"),
                bbox_inches="tight")
    plt.close()
    print("   fig_10")

    # Figure 11 — satisfaction vs late rate scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (name, grp) in enumerate(seller.groupby("segment_name")):
        ax.scatter(grp["late_rate"] * 100, grp["sat_rate"] * 100,
                   s=grp["order_count"] * 0.5, alpha=0.5,
                   color=PALETTE[i % len(PALETTE)], label=name,
                   edgecolors="black", linewidth=0.2)
    ax.set_xlabel("Late Delivery Rate (%)")
    ax.set_ylabel("Satisfaction Rate (%)")
    ax.set_title("Figure 11: Seller Satisfaction vs Late Rate\n"
                 "(bubble size = order count)")
    ax.legend(loc="lower left")
    ax.axhline(y=df["satisfied"].mean() * 100, color="grey",
               linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_11_seller_scatter.png"),
                bbox_inches="tight")
    plt.close()
    print("   fig_11")

    # Strategic recommendations
    recs = []
    for _, r in profile.iterrows():
        if r["sat_rate"] > 0.8 and r["late_rate"] < 0.1:
            strat = "Reward: featured placement, reduced commission"
        elif r["late_rate"] > 0.15:
            strat = "Improve: logistics support, delivery monitoring"
        elif r["sat_rate"] < profile["sat_rate"].median():
            strat = "Support: quality guidelines, feedback loops"
        else:
            strat = "Maintain: standard monitoring, periodic review"
        recs.append({
            "Segment": r.name, "Sellers": int(r["n"]),
            "Satisfaction": f"{r['sat_rate']:.1%}",
            "Late_Rate": f"{r['late_rate']:.1%}", "Strategy": strat,
        })
    pd.DataFrame(recs).to_csv(
        os.path.join(RES_DIR, "seller_recommendations.csv"), index=False)
    print("   seller_segments.csv, seller_recommendations.csv")


# ── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Olist E-Commerce — Analysis Pipeline\n")

    # Load prepared data
    df = pd.read_csv(MASTER_PATH, parse_dates=[
        "order_purchase_timestamp", "order_delivered_customer_date",
        "order_estimated_delivery_date", "order_delivered_carrier_date",
        "order_approved_at",
    ])
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["order_purchase_timestamp"])
    test_df  = pd.read_csv(TEST_PATH, parse_dates=["order_purchase_timestamp"])
    print(f"   Master: {len(df):,}  Train: {len(train_df):,}  Test: {len(test_df):,}")

    # Seller features (training data only — prevents leakage)
    train_df, test_df = add_seller_features(train_df, test_df)

    # Run all sections
    run_eda(df)
    run_corridor_analysis(df)
    run_prediction(train_df, test_df)
    run_seller_analysis(df)

    # Summary
    figs = [f for f in os.listdir(FIG_DIR) if f.endswith(".png")]
    csvs = [f for f in os.listdir(RES_DIR) if f.endswith(".csv")]
    print(f"\nComplete: {len(figs)} figures, {len(csvs)} result CSVs")
    print("Done.")
