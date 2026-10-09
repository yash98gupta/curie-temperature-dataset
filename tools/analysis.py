"""Single deterministic analysis run (seed 42) that produces every number and figure in the revised
manuscript, so that text, Table 1, captions, and figures agree.

Outputs:
  tools/results.json     all metrics referenced by the manuscript
  tools/figs_new/*.png   Figure 1 (a-b, c), Figure 2, Figure 3 (a, b, c, d), graphical abstract
"""
import os, json, time, warnings, hashlib, platform
from importlib.metadata import version
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import shap

SEED = 42
OUT = "tools/figs_new"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 220, "savefig.dpi": 220})
TORD = r"$T_{\mathrm{ord}}$"
BLUE, ORANGE, RED = "#3b7fb6", "#d9731a", "#e8604c"

df = pd.read_csv("curie_temperature_dataset.csv")
feat_cols = [c for c in df.columns if c.startswith("MagpieData")]
short = {c: c.replace("MagpieData ", "") for c in feat_cols}
X = df[feat_cols].values.astype(float)
T = df["tc"].values.astype(float)
y = np.log1p(T)
R = {"n": int(len(df)), "n_features": len(feat_cols), "n_classes": int(df.material_class.nunique()),
     "T_min": float(T.min()), "T_max": float(T.max()), "T_median": float(np.median(T)),
     "n_ge_300K": int((T >= 300).sum()), "n_ge_400K": int((T >= 400).sum()),
     "n_sources": int(df.reference.nunique()), "sources": df.reference.value_counts().to_dict()}


def make_models(seed):
    return {
        "XGBoost": XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                                random_state=seed, n_jobs=1, verbosity=0, tree_method="hist",
                                objective="reg:squarederror"),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                                       random_state=seed),
        "Random Forest": RandomForestRegressor(n_estimators=500, max_depth=15, random_state=seed, n_jobs=-1),
        "Ridge": Ridge(alpha=1.0, solver="svd"),
        "Lasso": Lasso(alpha=0.01, max_iter=50000, selection="cyclic"),
    }


def kmetrics(ylog_true, ylog_pred):
    t, p = np.expm1(ylog_true), np.expm1(ylog_pred)
    return dict(R2=r2_score(t, p), MAE=mean_absolute_error(t, p), RMSE=float(np.sqrt(mean_squared_error(t, p))))


def split(seed):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, shuffle=True)
    sc = StandardScaler().fit(X[tr])
    return tr, te, sc.transform(X[tr]), sc.transform(X[te])


# ------------------------------------------------------------------ main split, all models
tr, te, Xtr, Xte = split(SEED)
R["n_train"], R["n_test"] = int(len(tr)), int(len(te))
table = {}
models = make_models(SEED)
for name, m in models.items():
    m.fit(Xtr, y[tr])
    trm, tem = kmetrics(y[tr], m.predict(Xtr)), kmetrics(y[te], m.predict(Xte))
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cvk, cvl = [], []
    for a, b in kf.split(Xtr):
        fold_scaler = StandardScaler().fit(X[tr][a])
        mm = make_models(SEED)[name].fit(fold_scaler.transform(X[tr][a]), y[tr][a])
        pr = mm.predict(fold_scaler.transform(X[tr][b]))
        cvk.append(kmetrics(y[tr][b], pr)["R2"]); cvl.append(r2_score(y[tr][b], pr))
    table[name] = dict(train_R2=trm["R2"], test_R2=tem["R2"], MAE=tem["MAE"], RMSE=tem["RMSE"],
                       cv_log_mean=float(np.mean(cvl)), cv_log_std=float(np.std(cvl)),
                       cv_K_mean=float(np.mean(cvk)), cv_K_std=float(np.std(cvk)))
    table[name]["cv_K_folds"] = cvk
    table[name]["cv_log_folds"] = cvl
    print(f"{name:18s} " + " ".join(f"{k}={v:.3f}" for k, v in table[name].items() if not isinstance(v, list)), flush=True)
R["table1"] = table
xgb = models["XGBoost"]
pred_te = xgb.predict(Xte)
t_true, t_pred = T[te].copy(), np.expm1(pred_te)

# unscaled check (reviewer: why standardize trees?)
sc_free = make_models(SEED)["XGBoost"].fit(X[tr], y[tr])
R["xgb_unscaled_test"] = kmetrics(y[te], sc_free.predict(X[te]))

# ------------------------------------------------------------------ regime-wise errors on the test set
regimes = {"<200 K": t_true < 200, "200-400 K": (t_true >= 200) & (t_true < 400),
           "400-800 K": (t_true >= 400) & (t_true < 800), ">=800 K": t_true >= 800}
R["regime"] = {k: dict(n=int(v.sum()), MAE=float(mean_absolute_error(t_true[v], t_pred[v])),
                       bias=float(np.mean(t_pred[v] - t_true[v]))) for k, v in regimes.items() if v.sum()}
# screening-threshold check: of test materials predicted >= 500 K, fraction with measured >= 400 K
sel = t_pred >= 500
R["screen_pred500"] = dict(n=int(sel.sum()), frac_true_ge400=float(np.mean(t_true[sel] >= 400)),
                           frac_true_ge300=float(np.mean(t_true[sel] >= 300)))
sel3 = t_pred >= 400
R["screen_pred400"] = dict(n=int(sel3.sum()), frac_true_ge300=float(np.mean(t_true[sel3] >= 300)))

# ------------------------------------------------------------------ bootstrap (percentile, n = n_test, 2000 resamples)
rng = np.random.default_rng(SEED)
b_r2, b_mae = [], []
n = len(t_true)
for _ in range(2000):
    i = rng.integers(0, n, n)
    b_r2.append(r2_score(t_true[i], t_pred[i])); b_mae.append(mean_absolute_error(t_true[i], t_pred[i]))
R["bootstrap"] = dict(n_resamples=2000, n_per_resample=int(n), rng="numpy.random.default_rng(42)",
                      R2_lo=float(np.percentile(b_r2, 2.5)), R2_hi=float(np.percentile(b_r2, 97.5)),
                      MAE_lo=float(np.percentile(b_mae, 2.5)), MAE_hi=float(np.percentile(b_mae, 97.5)))
print("bootstrap", R["bootstrap"])

# ------------------------------------------------------------------ ten seeds
seeds = {}
for s in range(10):
    a, b, Xa, Xb = split(s)
    m = make_models(s)["XGBoost"].fit(Xa, y[a])
    seeds[s] = kmetrics(y[b], m.predict(Xb))
R["ten_seeds"] = dict(R2_mean=float(np.mean([v["R2"] for v in seeds.values()])),
                      R2_std=float(np.std([v["R2"] for v in seeds.values()])),
                      MAE_mean=float(np.mean([v["MAE"] for v in seeds.values()])),
                      MAE_std=float(np.std([v["MAE"] for v in seeds.values()])), per_seed=seeds)
print("ten seeds", {k: v for k, v in R["ten_seeds"].items() if k != "per_seed"})

# ------------------------------------------------------------------ SHAP (TreeSHAP on training portion)
expl = shap.TreeExplainer(xgb, feature_perturbation="tree_path_dependent", model_output="raw")
sv = expl.shap_values(Xtr)
mabs = np.abs(sv).mean(0)
order = np.argsort(mabs)[::-1]
R["shap_top"] = [(short[feat_cols[i]], float(mabs[i])) for i in order[:10]]
print("SHAP top:", R["shap_top"][:4])
i_mag = feat_cols.index("MagpieData mean GSmagmom")

# ------------------------------------------------------------------ learning curve (5-fold CV on training portion)
fracs = np.linspace(0.1, 1.0, 10)
lc = {"n": [], "train": [], "val": [], "val_std": []}
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for f in fracs:
    trs, vls = [], []
    for a, b in kf.split(Xtr):
        k = max(20, int(round(f * len(a))))
        sub = a[:k]
        subset_scaler = StandardScaler().fit(X[tr][sub])
        subset_train = subset_scaler.transform(X[tr][sub])
        subset_val = subset_scaler.transform(X[tr][b])
        m = make_models(SEED)["XGBoost"].fit(subset_train, y[tr][sub])
        trs.append(kmetrics(y[tr][sub], m.predict(subset_train))["R2"])
        vls.append(kmetrics(y[tr][b], m.predict(subset_val))["R2"])
    lc["n"].append(int(round(f * len(Xtr) * 0.8))); lc["train"].append(float(np.mean(trs)))
    lc["val"].append(float(np.mean(vls))); lc["val_std"].append(float(np.std(vls)))
R["learning_curve"] = lc
print("learning curve n:", lc["n"]); print("val R2:", np.round(lc["val"], 3))

# ------------------------------------------------------------------ cost of featurization (for CPU-hours comparison)
try:
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    ep = ElementProperty.from_preset("magpie")
    comps = [Composition(f) for f in df.formula]
    t0 = time.perf_counter()
    for c in comps[:200]:
        ep.featurize(c)
    R["featurize_ms_per_compound"] = (time.perf_counter() - t0) / 200 * 1e3
    t0 = time.perf_counter(); xgb.predict(Xte); R["predict_ms_per_compound"] = (time.perf_counter() - t0) / len(Xte) * 1e3
    print("featurize ms/compound", R["featurize_ms_per_compound"], "| predict ms/compound", R["predict_ms_per_compound"])
except Exception as e:
    print("featurize timing skipped:", e)

R["versions"] = {package: version(package) for package in ("numpy", "pandas", "scikit-learn", "xgboost", "shap", "matplotlib")}
R["versions"]["python"] = platform.python_version()
with open("curie_temperature_dataset.csv", "rb") as source:
    R["dataset_sha256"] = hashlib.sha256(source.read()).hexdigest()
R["validation"] = {"scaling": "fitted within each CV fold and learning-curve subset", "seed": SEED,
                   "ddof": 0, "learning_curve_fold_sizes": [len(indices) for indices, _ in kf.split(Xtr)]}
with open("tools/results.json", "w") as output:
    json.dump(R, output, indent=1, default=float)
xgb.save_model("tools/xgboost_seed42.json")
scaler = StandardScaler().fit(X[tr])
with open("tools/preprocessing.json", "w") as output:
    json.dump({"features": feat_cols, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
               "train_indices": tr.tolist(), "test_indices": te.tolist(), "target": "log1p(T_ord / K)"}, output, indent=1)
pd.DataFrame({"row_index": te, "formula": df.formula.iloc[te].values,
              "reported_K": t_true, "predicted_K": t_pred}).to_csv("tools/test_predictions.csv", index=False)

# ================================================================== FIGURES
def box(ax, s):
    ax.text(0.97, 0.95, s, transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="#fdf2dc", edgecolor="#8c7a4e"))

# Figure 1 (a,b)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
axes[0].hist(T, bins=40, color=BLUE, edgecolor="black", linewidth=0.5)
axes[0].set(title=f"(a) Raw {TORD} distribution", xlabel=f"Ordering temperature {TORD} (K)", ylabel="Number of materials")
box(axes[0], f"n = {len(T)}\nMedian = {np.median(T):.0f} K\nRange: {T.min():.0f}–{T.max():.0f} K")
axes[1].hist(y, bins=40, color=ORANGE, edgecolor="black", linewidth=0.5)
axes[1].set(title="(b) Log-transformed target", xlabel=f"ln(1 + {TORD}/K)", ylabel="Number of materials")
box(axes[1], f"Range: {y.min():.2f}–{y.max():.2f}")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1ab.png"); plt.close(fig)

# Figure 1 (c)
counts = df.material_class.value_counts()
fig, ax = plt.subplots(figsize=(8.5, 8.5))
cmap = plt.get_cmap("tab20")
ax.barh(range(len(counts)), counts.values[::-1], color=[cmap(i % 20) for i in range(len(counts))][::-1], edgecolor="black", linewidth=0.4)
ax.set_yticks(range(len(counts))); ax.set_yticklabels(counts.index[::-1], fontsize=8.5)
for i, v in enumerate(counts.values[::-1]):
    ax.text(v + 0.6, i, str(v), va="center", fontsize=7.5)
ax.set(xlabel="Number of compounds", title=f"(c) Dataset composition: {len(df)} materials, {len(counts)} classes", xlim=(0, counts.max() * 1.1))
fig.tight_layout(); fig.savefig(f"{OUT}/fig1c.png"); plt.close(fig)

# Figure 2
names = list(table)
lab = [n.replace(" ", "\n") for n in names]
test = [table[n]["test_R2"] for n in names]; cvL = [table[n]["cv_log_mean"] for n in names]; mae = [table[n]["MAE"] for n in names]
cols = [BLUE] * 3 + [RED] * 2
x = np.arange(len(names))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.bar(x, test, color=cols, edgecolor="black", linewidth=0.5, label="Held-out test $R^2$ (kelvin scale)")
for i, v in enumerate(test):
    ax.text(i, v - 0.06, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold", color="white")
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0, 1.0); ax.set_ylabel("$R^2$")
ax.set_title("(a) Coefficient of determination")
ax.annotate("", xy=(2.5, test[3]), xytext=(2.5, test[2]), arrowprops=dict(arrowstyle="<->", color="#c8500a", lw=1.8))
ax.text(2.5, (test[2] + test[3]) / 2 + 0.04, f"{test[2]-test[3]:.2f}\ngap", color="#c8500a", fontweight="bold", fontsize=9.5, ha="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none"))
ax.legend(loc="upper right", fontsize=8.5)
ax = axes[1]
ax.bar(x, mae, color=cols, edgecolor="black", linewidth=0.5)
for i, v in enumerate(mae):
    ax.text(i, v + 2.5, f"{v:.0f} K", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("Mean absolute error (K)"); ax.set_ylim(0, 215)
ax.set_title("(b) Mean absolute error on held-out test set")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2.png"); plt.close(fig)

# Figure 3a parity
fig, ax = plt.subplots(figsize=(5.6, 5.4))
scat = ax.scatter(t_true, t_pred, c=np.log10(np.abs(t_pred - t_true) + 1), cmap="viridis", s=26, edgecolor="black", linewidth=0.3)
lim = [0, 1450]; ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set(xlabel=f"Experimental {TORD} (K)", ylabel=f"Predicted {TORD} (K)", title="(a) XGBoost parity plot (test set)")
cb = fig.colorbar(scat, ax=ax, fraction=0.046, pad=0.03); cb.set_label("log$_{10}$(|error| + 1)")
box(ax, f"$R^2$ = {table['XGBoost']['test_R2']:.3f}\nMAE = {table['XGBoost']['MAE']:.0f} K\nRMSE = {table['XGBoost']['RMSE']:.0f} K\nn = {n}")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3a.png"); plt.close(fig)

# Figure 3b SHAP summary
import pandas as _pd
Xtr_df = _pd.DataFrame(Xtr, columns=[short[c] for c in feat_cols])
plt.figure(figsize=(7.2, 5.0))
shap.summary_plot(sv, Xtr_df, max_display=12, show=False, plot_size=None)
ax = plt.gca(); ax.set_xlabel(f"SHAP value (impact on ln(1 + {TORD}/K))")
plt.title("(b) SHAP feature importance (XGBoost, training portion)", fontsize=12)
plt.tight_layout(); plt.savefig(f"{OUT}/fig3b.png"); plt.close()

# Figure 3c SHAP dependence for mean GSmagmom
fig, ax = plt.subplots(figsize=(6.0, 4.4))
sc_ = ax.scatter(Xtr[:, i_mag], sv[:, i_mag], c=T[tr], cmap="plasma", s=22, edgecolor="black", linewidth=0.25)
cb = fig.colorbar(sc_, ax=ax, fraction=0.046, pad=0.03); cb.set_label(f"Experimental {TORD} (K)")
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set(xlabel="mean GSmagmom (standardized)", ylabel="SHAP value for mean GSmagmom", title="(c) SHAP dependence: mean GSmagmom")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3c.png"); plt.close(fig)

# Figure 3d learning curve
fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.plot(lc["n"], lc["train"], "o-", color=BLUE, label="Training $R^2$")
ax.plot(lc["n"], lc["val"], "s-", color=ORANGE, label="Validation $R^2$ (5-fold CV)")
ax.fill_between(lc["n"], np.array(lc["val"]) - np.array(lc["val_std"]), np.array(lc["val"]) + np.array(lc["val_std"]), color=ORANGE, alpha=0.2)
ax.set(xlabel="Number of training materials", ylabel="$R^2$ (kelvin scale)", title="(d) Learning curve (XGBoost)", ylim=(0, 1.05))
ax.legend(loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3d.png"); plt.close(fig)

# Graphical abstract: parity + top-5 SHAP bar
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
ax = axes[0]
ax.scatter(t_true, t_pred, color=BLUE, s=24, edgecolor="black", linewidth=0.3)
ax.plot(lim, lim, "k--", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
ax.axhspan(0, 400, color="gray", alpha=0.08)
ax.text(1420, 415, "400 K reference (not device qualification)", fontsize=8.5, color="gray", ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
ax.set(xlabel=f"Reported {TORD} (K)", ylabel=f"Predicted {TORD} (K)", title="Composition-only XGBoost: 134 test materials")
box(ax, f"$R^2$ = {table['XGBoost']['test_R2']:.3f}\nMAE = {table['XGBoost']['MAE']:.0f} K")
ax = axes[1]
top5 = [(short[feat_cols[i]], mabs[i]) for i in order[:5]][::-1]
ax.barh([t[0] for t in top5], [t[1] for t in top5], color=[ORANGE if "GSmagmom" in t[0] else "#9bb7d4" for t in top5], edgecolor="black", linewidth=0.4)
ax.set(xlabel="mean |SHAP| (log-target scale)", title="Elemental magnetic-moment descriptor\nhas the largest mean absolute attribution")
fig.suptitle(f"Machine learning prediction of magnetic ordering temperatures {TORD}", fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/graphical_abstract.png"); plt.close(fig)
print("figures written to", OUT)
