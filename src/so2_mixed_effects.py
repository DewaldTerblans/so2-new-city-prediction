"""
STTN327 SO2 project - Mixed-effects model (random intercept per city)
Reproduces the numbers in SO2_Group_Starter_Doc.docx.
SO2_ij = b0 + b1*temp + b2*wind + b3*precip + b4*precipdays + b5*manuf + b6*pop
         + g2*Q2 + g3*Q3 + g4*Q4 + u_i + e_ij,  u_i ~ N(0,tau2), e_ij ~ N(0,sigma2)
Fitted by REML (statsmodels MixedLM).  Evaluation: leave-one-city-out (LOCO).
Run:  python so2_mixed_effects.py [path/to/quarterly_SO2_data.csv]
Needs: pandas numpy statsmodels scipy scikit-learn
"""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "quarterly_SO2_data/quarterly_SO2_data.csv"
d = pd.read_csv(CSV)
d["quarter"] = d["quarter"].astype(int)
d = d.sort_values(["city", "year", "quarter"]).reset_index(drop=True)
d["t"] = d.groupby("city").cumcount()          # 0..11 time order within city
d["Q2"] = (d.quarter == 2).astype(float); d["Q3"] = (d.quarter == 3).astype(float); d["Q4"] = (d.quarter == 4).astype(float)
XCOLS = ["temperature", "windspeed", "precipitation", "precipitationdays",
         "manufacturing_enterprises", "population_thousands", "Q2", "Q3", "Q4"]
FORMULA = "SO2 ~ " + " + ".join(XCOLS)
cities = list(d.city.unique())


def fit(data):
    m = smf.mixedlm(FORMULA, data, groups=data["city"]).fit(reml=True)
    return m


def index_of(m, data):
    """SO2 Index = fixed part only (no city effect)."""
    b = m.fe_params
    return b["Intercept"] + data[XCOLS].values @ b[XCOLS].values


# ---------------------------------------------------------------- 1. full fit
full = fit(d)
tau2, sigma2 = float(full.cov_re.iloc[0, 0]), float(full.scale)
print("=== Mixed-effects fit (all 41 cities) ===")
print(full.fe_params.round(3).to_string())
print(f"tau2 (between-city) = {tau2:.2f}   sigma2 (within-city) = {sigma2:.2f}   ICC = {tau2/(tau2+sigma2):.3f}")
fe_var = np.var(d[XCOLS].values @ full.fe_params[XCOLS].values, ddof=1)
print(f"Nakagawa R2: marginal = {fe_var/(fe_var+tau2+sigma2):.3f}  conditional = {(fe_var+tau2)/(fe_var+tau2+sigma2):.3f}")

# ---------------------------------------------------------------- 2. LOCO
rows = []      # one row per held-out quarter
for c in cities:
    tr, te = d[d.city != c], d[d.city == c].copy()
    m = fit(tr)
    t2, s2 = float(m.cov_re.iloc[0, 0]), float(m.scale)
    idx = index_of(m, te)
    te["index"] = idx
    te["naive"] = tr.SO2.mean()
    for k in (0, 1, 2, 3):
        if k == 0:
            pred = idx.copy()
        else:
            miss = (te.SO2.values[:k] - idx[:k]).mean()      # average miss on revealed quarters
            lam = t2 * k / (t2 * k + s2)                      # partial pooling (empirical Bayes)
            pred = idx + lam * miss
        te[f"pred{k}"] = pred
    rows.append(te)
L = pd.concat(rows)


def pooled(df, col, k):
    sub = df[df.t >= k]                                       # score only the remaining quarters
    err = sub.SO2 - sub[col]
    naive_err = sub.SO2 - sub["naive"]
    rmse = np.sqrt((err**2).mean())
    r2 = 1 - (err**2).sum() / (naive_err**2).sum()            # pooled R2 vs naive guess
    per_city = sub.assign(e2=err**2).groupby("city").e2.mean().pow(0.5).mean()
    return rmse, r2, per_city


print("\n=== Leave-one-city-out results ===")
nrm, _, ncity = pooled(L, "naive", 0)
print(f"Naive guess (mean of other 40 cities): pooled RMSE {nrm:.2f}, avg per-city RMSE {ncity:.2f}")
for k in (0, 1, 2, 3):
    rmse, r2, pc = pooled(L, f"pred{k}", k)
    print(f"k={k}: pooled RMSE {rmse:.2f}  pooled R2 {r2:.3f}  avg per-city RMSE {pc:.2f}")

# ---------------------------------------------------------------- 3. conformal band (new city, k=0)
L["miss"] = L.SO2 - L["index"]
g = L.groupby("city").miss
n, alpha = len(cities), 0.10
j = int(np.ceil((n + 1) * (1 - alpha)))                       # = 38
abs_scores = np.sort(g.apply(lambda x: x.abs().max()).values)
up = np.sort(g.max().values)[j - 1]                           # upper margin
lo = np.sort((-g.min()).values)[j - 1]                        # lower margin
print(f"\n=== Conformal band (n={n}, alpha={alpha}, j={j}) ===")
print(f"Symmetric band: Index +/- {abs_scores[j-1]:.2f}")
print(f"One-sided margins (38th smallest signed score): upper {up:.2f}, lower {lo:.2f}")
print('NOTE: the starter doc quotes 23.62 / 21.54 for the threshold rules; those two values are not reproduced by this script.')
idx_full = pd.Series(index_of(full, d), index=d.index)
city_idx = idx_full.groupby(d.city).agg(["min", "max"])
print("Cities with Index <= T-upper (T=40) i.e. <=", round(40-up, 2), ":", int((city_idx["max"] <= 40 - up).sum()))
print("Cities with Index >= T+lower (T=20) i.e. >=", round(20+lo, 2), ":", int((city_idx["min"] >= 20 + lo).sum()))

# high/low classifier: city above/below median city mean SO2
cm = d.groupby("city").SO2.mean(); med = cm.median()
ci = L.groupby("city")["index"].mean()
y = (cm.loc[ci.index] > med).astype(int)
print(f"Median city SO2 = {med:.1f};  AUC = {roc_auc_score(y, ci):.2f};  accuracy (Index > {med:.1f}) = {((ci > med).astype(int) == y).mean():.2f}")

# ---------------------------------------------------------------- 4. diagnostics
print("\n=== Diagnostics ===")
res = full.resid
d["resid"] = res.values
lagged = d.groupby("city").resid.shift(1); ok = lagged.notna()
lag = np.corrcoef(d.resid[ok], lagged[ok])[0, 1]
dw = (d.groupby("city").resid.diff()**2).sum() / (d.resid**2).sum()
print(f"Lag-1 correlation within city: {lag:.2f};  Durbin-Watson: {dw:.2f}")
print(f"Residual skew {stats.skew(res):.2f}, Shapiro-Wilk p = {stats.shapiro(res)[1]:.3f}")
groups = [x.values for _, x in d.groupby("city").resid]
print(f"Residual SD per city: {min(np.std(x, ddof=1) for x in groups):.2f}-{max(np.std(x, ddof=1) for x in groups):.2f}; Levene p = {stats.levene(*groups)[1]:.4f}")
re = pd.Series({c: v.iloc[0] for c, v in full.random_effects.items()})
print(f"City intercepts: Shapiro p = {stats.shapiro(re)[1]:.3f}; largest: {re.sort_values().tail(3).round(1).to_dict()}")
X = d[XCOLS].assign(const=1.0)
print("VIF:", {c: round(float(variance_inflation_factor(X.values, i)), 1) for i, c in enumerate(X.columns) if c in ('manufacturing_enterprises', 'population_thousands')})
print("corr(manufacturing, population) =", round(d.manufacturing_enterprises.corr(d.population_thousands), 2))
miss_city = (L.groupby("city").miss.mean()).sort_values()
print("Biggest under-predictions (city SO2 above Index):", miss_city.tail(3).round(1).to_dict())
