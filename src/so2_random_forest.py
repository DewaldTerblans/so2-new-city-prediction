"""
STTN327 SO2 project - RANDOM FOREST (new-city prediction)
=========================================================
Run:  python so2_random_forest.py quarterly_SO2_data.csv rf_output
Needs: pandas numpy scikit-learn statsmodels scipy matplotlib   (+ so2_eval.py in the same folder)

What this script does (each part prints its own results and saves figures/CSVs to rf_output/):
  1. Why leave-one-city-out?  OOB / random-row CV / leave-one-city-out R2 for the same forest.
  2. Baseline forest (default settings), with and without windspeed (the Memphis/Charleston check).
  3. NESTED tuning: hyper-parameters are chosen by city-grouped cross-validation INSIDE the
     40 training cities of every leave-one-city-out fold, so the held-out city is never used to tune.
  4. Final scores for k = 0, 1, 2 with the shared protocol (so2_eval.py).
  5. Comparison with reference runs of the other models + city-level bootstrap confidence intervals.
  6. Variable importance: grouped permutation importance (handles manufacturing/population r = 0.96).
  7. Partial dependence and a failure analysis (which cities does the forest get wrong, and why).
"""
import sys, os, time, pickle, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import statsmodels.formula.api as smf
from so2_eval import load_data, loco_predict, calibrate, score, FEATURES

warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "quarterly_SO2_data.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "rf_output"
os.makedirs(OUT, exist_ok=True)
SEED = 2026
RF_FEATURES = [f for f in FEATURES if f != "windspeed"]   # windspeed dropped: Memphis/Charleston values are impossible
GRID = [(leaf, mtry) for leaf in (1, 3, 5, 10, 20) for mtry in (2, 3, 5)]   # (min_samples_leaf, max_features)
INNER_TREES, FINAL_TREES = 100, 500
CACHE = os.path.join(OUT, "nested_cache.pkl")

d = load_data(CSV)
cities = list(d.city.unique())
t0 = time.time()


def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)


def rf(**kw):
    return RandomForestRegressor(random_state=SEED, n_jobs=-1, **kw)


# ------------------------------------------------------------------ plot style (dataviz skill)
SURFACE, INK, INK2, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#9c9b96"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRIDC, "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
                     "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=200); plt.close(fig); log("saved", name)


# ================================================================== 1. why leave-one-city-out
log("1. leakage demonstration")
X, y, g = d[RF_FEATURES].values, d.SO2.values, d.city.values
m_oob = rf(n_estimators=500, oob_score=True).fit(X, y)
r2_oob = m_oob.oob_score_
# random 5-fold by ROW: quarters of the same city land in train and test
e_row, e_nv_row = [], []
for tr, te in KFold(5, shuffle=True, random_state=SEED).split(X):
    m = rf(n_estimators=300).fit(X[tr], y[tr])
    e_row += list(y[te] - m.predict(X[te])); e_nv_row += list(y[te] - y[tr].mean())
r2_row = 1 - np.sum(np.square(e_row)) / np.sum(np.square(e_nv_row))
L_def = loco_predict(lambda tr, te: rf(n_estimators=500).fit(tr[RF_FEATURES], tr.SO2).predict(te[RF_FEATURES]), d)
r2_loco_default = score(L_def, ks=(0,)).pooled_R2.iloc[0]
print(f"   Out-of-bag R2 (rows)      : {r2_oob:.3f}\n   Random 5-fold by row R2   : {r2_row:.3f}\n   Leave-one-city-out R2     : {r2_loco_default:.3f}")

# ================================================================== 2. baseline + windspeed sensitivity
log("2. baseline forests")
base_rows = []
for label, cols in (("RF default, with windspeed", FEATURES), ("RF default, no windspeed", RF_FEATURES)):
    Lb = loco_predict(lambda tr, te, c=cols: rf(n_estimators=500).fit(tr[c], tr.SO2).predict(te[c]), d)
    s = score(Lb); s.insert(0, "model", label); base_rows.append(s.reset_index())
baseline = pd.concat(base_rows)
print(baseline.to_string(index=False))


# ================================================================== 3. nested tuning
def select_params(train, return_scores=False):
    """Choose (min_samples_leaf, max_features) by city-grouped 5-fold CV inside `train` only."""
    Xt, yt, gt = train[RF_FEATURES].values, train.SO2.values, train.city.values
    folds = list(GroupKFold(5).split(Xt, yt, gt))
    scores = {}
    for leaf, mtry in GRID:
        e2 = []
        for tr_i, va_i in folds:
            m = rf(n_estimators=INNER_TREES, min_samples_leaf=leaf, max_features=mtry).fit(Xt[tr_i], yt[tr_i])
            e2 += list((yt[va_i] - m.predict(Xt[va_i])) ** 2)
        scores[(leaf, mtry)] = float(np.sqrt(np.mean(e2)))
    best = min(scores, key=lambda k: (round(scores[k], 6), -k[0]))          # tie-break: simpler (bigger leaf)
    return (best, scores) if return_scores else best


if os.path.exists(CACHE):
    log("3. loading cached nested-CV run", CACHE)
    L_tuned, chosen = pickle.load(open(CACHE, "rb"))
else:
    log("3. nested tuning: 41 outer folds x", len(GRID), "settings x 5 inner folds (about 10 minutes)")
    chosen = []

    def tuned_fit_predict(tr, te):
        leaf, mtry = select_params(tr)
        chosen.append({"held_out_city": te.city.iloc[0], "min_samples_leaf": leaf, "max_features": mtry})
        log(f"   {te.city.iloc[0]:24s} chose leaf={leaf:2d}, max_features={mtry}")
        return rf(n_estimators=FINAL_TREES, min_samples_leaf=leaf, max_features=mtry).fit(tr[RF_FEATURES], tr.SO2).predict(te[RF_FEATURES])

    L_tuned = loco_predict(tuned_fit_predict, d)
    pickle.dump((L_tuned, chosen), open(CACHE, "wb"))

chosen = pd.DataFrame(chosen)
chosen.to_csv(os.path.join(OUT, "rf_nested_choices.csv"), index=False)
freq = chosen.groupby(["min_samples_leaf", "max_features"]).size().sort_values(ascending=False)
print("   Settings chosen across the 41 outer folds:\n", freq.to_string())

(best, grid_scores) = select_params(d, return_scores=True)           # final settings, tuned on all 41 cities
FINAL = dict(min_samples_leaf=best[0], max_features=best[1])
print(f"   FINAL settings (grouped 5-fold CV on all 41 cities): {FINAL}  inner CV RMSE = {grid_scores[best]:.2f}")
pd.DataFrame([{"min_samples_leaf": k[0], "max_features": k[1], "inner_cv_rmse": v} for k, v in grid_scores.items()]
             ).to_csv(os.path.join(OUT, "rf_tuning_grid.csv"), index=False)

leaves = sorted({k[0] for k in GRID}); mtrys = sorted({k[1] for k in GRID})
M = np.array([[grid_scores[(l, m)] for m in mtrys] for l in leaves])
fig, ax = plt.subplots(figsize=(7.2, 4.8))
im = ax.imshow(M, cmap="Blues_r", aspect="auto")
ax.set_xticks(range(len(mtrys))); ax.set_xticklabels(mtrys); ax.set_yticks(range(len(leaves))); ax.set_yticklabels(leaves)
ax.set_xlabel("max_features (features tried per split)"); ax.set_ylabel("min_samples_leaf"); ax.grid(False)
for i in range(len(leaves)):
    for j in range(len(mtrys)):
        ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", color="white" if M[i, j] < np.median(M) else INK,
                fontweight="bold" if (leaves[i], mtrys[j]) == best else "normal")
ax.set_title("Tuning grid: city-grouped CV RMSE (darker = better)")
save(fig, "fig2_tuning_grid.png")

# ================================================================== 4. final scores
log("4. final scores")
tuned = score(L_tuned); tuned.insert(0, "model", "RF tuned (nested CV), no windspeed")
naive_row = {"model": "Naive guess (mean of other 40 cities)", "k": 0, "pooled_RMSE": round(float(np.sqrt(np.mean((L_tuned.SO2 - L_tuned.naive) ** 2))), 3), "pooled_R2": 0.0, "avg_per_city_RMSE": np.nan}
results = pd.concat([pd.DataFrame([naive_row]), baseline, tuned.reset_index()])
results.to_csv(os.path.join(OUT, "rf_results_table.csv"), index=False)
print(results.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4.8))
labels = ["Out-of-bag\n(rows)", "Random 5-fold\n(rows)", "Leave-one-city-out\n(default forest)", "Leave-one-city-out\n(tuned, nested)"]
vals = [r2_oob, r2_row, r2_loco_default, tuned.loc[0].pooled_R2]
bars = ax.bar(labels, vals, color=[MUTED, MUTED, BLUE, BLUE], width=0.55)
for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width() / 2, max(v, 0) + 0.02, f"{v:.2f}", ha="center", color=INK, fontweight="bold")
ax.set_ylim(min(0, min(vals)) - 0.05, 1.08); ax.set_ylabel("R² (vs naive guess)")
ax.set_title("Same forest, very different score: only the new-city scores are honest"); ax.grid(axis="x", visible=False)
save(fig, "fig1_why_leave_one_city_out.png")

# ================================================================== 5. comparison + bootstrap
log("5. reference runs of the other models + city bootstrap")


def mixed_fp(tr, te):
    m = smf.mixedlm("SO2 ~ " + " + ".join(FEATURES), tr, groups=tr.city).fit(reml=True)
    b = m.fe_params; return b["Intercept"] + te[FEATURES].values @ b[FEATURES].values


mm = smf.mixedlm("SO2 ~ " + " + ".join(FEATURES), d, groups=d.city).fit(reml=True)
t2, s2 = float(mm.cov_re.iloc[0, 0]), float(mm.scale)
mixed_lam = lambda k, data: t2 * k / (t2 * k + s2)


def sk(model): return lambda tr, te: model.fit(tr[FEATURES], tr.SO2).predict(te[FEATURES])


refs = {"Mixed-effects": (loco_predict(mixed_fp, d), mixed_lam),
        "Linear regression": (loco_predict(sk(LinearRegression()), d), None),
        "Ridge": (loco_predict(sk(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))), d), None),
        "Lasso": (loco_predict(sk(make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=20000))), d), None),
        "Random forest (tuned)": (L_tuned, None)}
KS = (0, 1, 2)


def city_stats(L, k, lam):
    sub = calibrate(L, k, 1.0 if k == 0 else (lam(k, d) if lam else 1.0))
    e2 = (sub.SO2 - sub.pred) ** 2; n2 = (sub.SO2 - sub.naive) ** 2
    return (e2.groupby(sub.city).sum().reindex(cities).values, n2.groupby(sub.city).sum().reindex(cities).values,
            sub.groupby("city").size().reindex(cities).values)


rng = np.random.default_rng(SEED); B = 5000
IDX = rng.integers(0, len(cities), (B, len(cities)))                  # resample whole cities (the independent units)
boot, point = {}, {}
for name, (L, lam) in refs.items():
    for k in KS:
        sse, sst, n = city_stats(L, k, lam)
        rm = np.sqrt(sse[IDX].sum(1) / n[IDX].sum(1)); r2 = 1 - sse[IDX].sum(1) / sst[IDX].sum(1)
        boot[(name, k)] = (rm, r2); point[(name, k)] = (np.sqrt(sse.sum() / n.sum()), 1 - sse.sum() / sst.sum())
sse0, sst0, n0 = city_stats(L_tuned, 0, None)
naive_boot = np.sqrt(sst0[IDX].sum(1) / n0[IDX].sum(1)); naive_pt = np.sqrt(sst0.sum() / n0.sum())
rows = []
for (name, k), (rm, r2) in boot.items():
    rows.append({"model": name, "k": k, "RMSE": point[(name, k)][0], "RMSE_lo": np.percentile(rm, 2.5), "RMSE_hi": np.percentile(rm, 97.5),
                 "R2": point[(name, k)][1], "R2_lo": np.percentile(r2, 2.5), "R2_hi": np.percentile(r2, 97.5)})
comp = pd.DataFrame(rows).round(3); comp.to_csv(os.path.join(OUT, "rf_model_comparison_bootstrap.csv"), index=False)
print("   (reference re-implementations; the group's own model numbers go in the final table)")
print(comp.to_string(index=False))

print("\n   Paired difference in pooled RMSE, RF minus other model (positive = RF worse); 95% city-bootstrap CI")
diffs = []
for other in ("Mixed-effects", "Linear regression", "Ridge", "Lasso"):
    for k in KS:
        dlt = boot[("Random forest (tuned)", k)][0] - boot[(other, k)][0]
        diffs.append({"vs": other, "k": k, "RMSE_diff": point[("Random forest (tuned)", k)][0] - point[(other, k)][0],
                      "lo": np.percentile(dlt, 2.5), "hi": np.percentile(dlt, 97.5), "P(RF worse)": float((dlt > 0).mean())})
diffs = pd.DataFrame(diffs).round(3); diffs.to_csv(os.path.join(OUT, "rf_paired_differences.csv"), index=False)
print(diffs.to_string(index=False))

# figure 6: k = 0 pooled RMSE with 95% CI
order = [("Naive guess", naive_pt, np.percentile(naive_boot, 2.5), np.percentile(naive_boot, 97.5))]
for name in refs:
    c = comp[(comp.model == name) & (comp.k == 0)].iloc[0]; order.append((name, c.RMSE, c.RMSE_lo, c.RMSE_hi))
order.sort(key=lambda r: r[1])
fig, ax = plt.subplots(figsize=(8.4, 4.8))
for i, (name, v, lo, hi) in enumerate(order):
    col = BLUE if name.startswith("Random") else (ORANGE if name.startswith("Mixed") else MUTED)
    ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round"); ax.plot(v, i, "o", color=col, ms=9, mec=SURFACE, mew=2)
    ax.text(hi + 0.5, i, f"{v:.1f}", va="center", color=INK, fontweight="bold")
ax.set_yticks(range(len(order))); ax.set_yticklabels([o[0] for o in order]); ax.invert_yaxis()
ax.set_xlabel("Pooled RMSE for a brand-new city (µg/m³), 95% CI from resampling cities"); ax.grid(axis="y", visible=False)
ax.set_title("Predicting SO2 in a new city (k = 0): lower is better"); ax.set_xlim(8, order[-1][3] + 4)
save(fig, "fig6_model_comparison_k0.png")

# ================================================================== 6. permutation importance (grouped)
log("6. permutation importance")
BLOCKS = {"City size & industry (manufacturing + population)": ["manufacturing_enterprises", "population_thousands"],
          "  manufacturing alone": ["manufacturing_enterprises"], "  population alone": ["population_thousands"],
          "Temperature": ["temperature"], "Precipitation (amount + days)": ["precipitation", "precipitationdays"],
          "Season (quarter)": ["Q2", "Q3", "Q4"]}
prng = np.random.default_rng(SEED); NREP = 30
imp = {k: [] for k in BLOCKS}
for tr_i, te_i in GroupKFold(10).split(X, y, g):
    m = rf(n_estimators=300, **FINAL).fit(X[tr_i], y[tr_i])
    Xte, yte = X[te_i].copy(), y[te_i]
    base = np.sqrt(np.mean((yte - m.predict(Xte)) ** 2))
    for name, cols in BLOCKS.items():
        ci = [RF_FEATURES.index(c) for c in cols]; inc = []
        for _ in range(NREP):
            Xp = Xte.copy(); perm = prng.permutation(len(Xp)); Xp[:, ci] = Xte[perm][:, ci]     # permute the block jointly
            inc.append(np.sqrt(np.mean((yte - m.predict(Xp)) ** 2)) - base)
        imp[name].append(np.mean(inc))
imp_df = pd.DataFrame([{"feature_or_block": k.strip(), "rmse_increase_mean": np.mean(v), "rmse_increase_sd_over_folds": np.std(v, ddof=1)} for k, v in imp.items()]).round(3)
m_all = rf(n_estimators=FINAL_TREES, **FINAL).fit(X, y)
imp_df.to_csv(os.path.join(OUT, "rf_permutation_importance.csv"), index=False)
print(imp_df.to_string(index=False))
print("   Impurity importance (all 41 cities; biased towards high-variance features):")
print(pd.Series(m_all.feature_importances_, RF_FEATURES).sort_values(ascending=False).round(3).to_string())

sel = imp_df.sort_values("rmse_increase_mean")
fig, ax = plt.subplots(figsize=(9, 4.8))
colors = [BLUE if "City size" in n or n in ("manufacturing alone", "population alone") else MUTED for n in sel.feature_or_block]
ax.barh([n.replace(" (manufacturing + population)", "\n(manufacturing + population)") for n in sel.feature_or_block],
        sel.rmse_increase_mean, xerr=sel.rmse_increase_sd_over_folds, color=colors, height=0.55,
        error_kw=dict(ecolor=INK2, lw=1, capsize=3))
for i, v in enumerate(sel.rmse_increase_mean): ax.text(max(v, 0) + 0.6, i + 0.28, f"{max(v, 0):+.1f}" if abs(v) >= 0.05 else "0.0", color=INK, fontsize=10, fontweight="bold", va="center")
ax.set_xlabel("Increase in held-out RMSE (µg/m³) when the feature is shuffled"); ax.grid(axis="y", visible=False)
ax.set_title("What the forest relies on\n(permutation importance, city-grouped folds)", fontsize=12)
save(fig, "fig3_permutation_importance.png")

# ================================================================== 7. partial dependence + failure analysis
log("7. partial dependence and failure analysis")
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
for ax, feat, lab in zip(axes, ["manufacturing_enterprises", "population_thousands", "temperature"],
                         ["Manufacturing enterprises (20+ workers)", "Population (thousands)", "Temperature (°F)"]):
    pdp = partial_dependence(m_all, d[RF_FEATURES], [feat], grid_resolution=60)
    ax.plot(pdp["grid_values"][0], pdp["average"][0], color=BLUE, lw=2)
    ax.plot(d[feat], np.full(len(d), ax.get_ylim()[0]), "|", color=MUTED, alpha=0.4, ms=8)
    ax.set_xlabel(lab); ax.set_title("")
axes[0].set_ylabel("Predicted SO2 (µg/m³)"); fig.suptitle("Partial dependence: flat beyond the data the forest has seen", x=0.01, ha="left", fontweight="bold")
save(fig, "fig7_partial_dependence.png")

cm = L_tuned.groupby("city").agg(actual=("SO2", "mean"), predicted=("base", "mean"),
                                 manufacturing=("manufacturing_enterprises", "first"), population=("population_thousands", "first"))
cm["error"] = cm.actual - cm.predicted
cm.sort_values("error").round(2).to_csv(os.path.join(OUT, "rf_city_errors_k0.csv"))
print("   Largest k=0 errors (actual - predicted, city mean):")
print(cm.reindex(cm.error.abs().sort_values(ascending=False).index).head(8).round(1).to_string())

fig, ax = plt.subplots(figsize=(7, 6))
lim = [0, max(cm.actual.max(), cm.predicted.max()) + 5]
ax.plot(lim, lim, color=MUTED, lw=1.2, ls="--"); ax.scatter(cm.predicted, cm.actual, s=45, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
for c, r in cm.iterrows():
    if abs(r.error) > 22: ax.annotate(c, (r.predicted, r.actual), xytext=(6, -13) if c == "Dallas" else (6, 4), textcoords="offset points", fontsize=9, color=INK)
ax.set_xlabel("Forest prediction for the city (never saw it in training), µg/m³"); ax.set_ylabel("Actual city mean SO2, µg/m³")
ax.set_title("Where the forest misses: a few heavy-SO2 cities"); ax.set_xlim(lim); ax.set_ylim(lim)
save(fig, "fig4_predicted_vs_actual.png")

es = cm.sort_values("error")
fig, ax = plt.subplots(figsize=(8.6, 9))
ax.barh(es.index, es.error, color=[BLUE if v >= 0 else ORANGE for v in es.error], height=0.65)
ax.axvline(0, color=INK2, lw=0.8); ax.set_xlabel("Actual minus predicted city mean SO2 (µg/m³)"); ax.grid(axis="y", visible=False)
ax.set_title("City-level error, k = 0 (blue: forest too low, orange: too high)"); ax.tick_params(axis="y", labelsize=8)
save(fig, "fig5_city_errors.png")

# ================================================================== 8. one-line summary
r0 = tuned.loc[0]
print(f"\nSUMMARY  RF tuned (nested CV), new city k=0: pooled RMSE {r0.pooled_RMSE:.2f}, pooled R2 {r0.pooled_R2:.3f}"
      f"  (naive guess {naive_row['pooled_RMSE']:.2f})   k=1: {tuned.loc[1].pooled_RMSE:.2f}   k=2: {tuned.loc[2].pooled_RMSE:.2f}")
log("done. Files in", OUT)
