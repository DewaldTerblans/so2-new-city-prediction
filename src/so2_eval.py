"""
STTN327 SO2 project - SHARED evaluation (leave-one-city-out, k = 0, 1, 2)
Everyone scores their model with this, so the comparison table is fair.
Usage in your own notebook/script:
    from so2_eval import load_data, evaluate
    d = load_data("quarterly_SO2_data.csv")
    def my_model(train, test):
        # fit on `train` (40 cities), return predictions (numpy array) for every row of `test`
        ...
        return preds
    print(evaluate(my_model, d))          # -> table with k = 0, 1, 2
Protocol (from the group starter doc):
  k = 0 : predict the left-out city from its weather/season/manufacturing/population only.
  k = 1,2: reveal that city's first k quarters of SO2, add the average miss on those
           quarters to the remaining predictions, score ONLY the remaining quarters.
  Pooled RMSE, and pooled R2 = 1 - SSE(model) / SSE(mean SO2 of the other 40 cities).
Run `python so2_eval.py quarterly_SO2_data.csv` for examples (OLS, Ridge, Lasso, random forest).
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

FEATURES = ["temperature", "windspeed", "precipitation", "precipitationdays",
            "manufacturing_enterprises", "population_thousands", "Q2", "Q3", "Q4"]


def load_data(path="quarterly_SO2_data.csv"):
    import os
    if not os.path.exists(path):   # also look in the quarterly_SO2_data/ sub-folder
        alt = os.path.join("quarterly_SO2_data", os.path.basename(path))
        path = alt if os.path.exists(alt) else path
    d = pd.read_csv(path)
    d["quarter"] = d["quarter"].astype(int)
    d = d.sort_values(["city", "year", "quarter"]).reset_index(drop=True)
    d["t"] = d.groupby("city").cumcount()          # 0..11 within city
    for q in (2, 3, 4):
        d[f"Q{q}"] = (d.quarter == q).astype(float)
    return d


def loco_predict(fit_predict, data, verbose=False):
    """Leave-one-city-out base predictions. Returns one row per quarter with columns base, naive."""
    out = []
    for c in data.city.unique():
        tr, te = data[data.city != c], data[data.city == c].copy()
        te["base"] = np.asarray(fit_predict(tr, te), dtype=float)
        te["naive"] = tr.SO2.mean()
        out.append(te)
        if verbose: print("done", c, flush=True)
    return pd.concat(out)


def calibrate(L, k, lam=1.0):
    """Add (lam x) the average miss on the first k quarters; keep only the remaining quarters."""
    sub = L.copy()
    if k > 0:
        miss = sub[sub.t < k].groupby("city").apply(lambda g: (g.SO2 - g.base).mean())
        sub["pred"] = sub.base + lam * sub.city.map(miss)
    else:
        sub["pred"] = sub.base
    return sub[sub.t >= k]


def score(L, ks=(0, 1, 2), shrink=None, data=None):
    rows = []
    for k in ks:
        lam = 1.0 if (shrink is None or k == 0) else shrink(k, data)
        sub = calibrate(L, k, lam)
        e, n = sub.SO2 - sub.pred, sub.SO2 - sub.naive
        per_city = (e**2).groupby(sub.city).mean().pow(0.5).mean()
        rows.append({"k": k, "pooled_RMSE": np.sqrt((e**2).mean()),
                     "pooled_R2": 1 - (e**2).sum() / (n**2).sum(), "avg_per_city_RMSE": per_city})
    return pd.DataFrame(rows).round(3).set_index("k")


def evaluate(fit_predict, data, ks=(0, 1, 2), shrink=None, verbose=False):
    """fit_predict(train_df, test_df) -> predictions for test_df rows.
    shrink(k, data) -> weight on the average miss (default 1.0 = add the full miss)."""
    return score(loco_predict(fit_predict, data, verbose), ks, shrink, data)


def naive_baseline(data):
    e = np.concatenate([d_c.SO2.values - data[data.city != c].SO2.mean()
                        for c, d_c in data.groupby("city")])
    return np.sqrt((e**2).mean())


# ----------------------------------------------------------------- examples
if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestRegressor
    import statsmodels.formula.api as smf

    d = load_data(sys.argv[1] if len(sys.argv) > 1 else "quarterly_SO2_data.csv")
    print(f"Naive baseline pooled RMSE (should be 23.98): {naive_baseline(d):.2f}\n")

    def sk(model):
        return lambda tr, te: model.fit(tr[FEATURES], tr.SO2).predict(te[FEATURES])

    def mixed(tr, te):
        m = smf.mixedlm("SO2 ~ " + " + ".join(FEATURES), tr, groups=tr.city).fit(reml=True)
        b = m.fe_params
        return b["Intercept"] + te[FEATURES].values @ b[FEATURES].values

    def mixed_shrink(k, data):                          # empirical-Bayes weight used in the doc
        m = smf.mixedlm("SO2 ~ " + " + ".join(FEATURES), data, groups=data.city).fit(reml=True)
        t2, s2 = float(m.cov_re.iloc[0, 0]), float(m.scale)
        return t2 * k / (t2 * k + s2)

    print("Mixed-effects (should give k=0 RMSE 15.44, R2 0.586):\n", evaluate(mixed, d, shrink=mixed_shrink), "\n")
    print("Multiple linear regression (OLS):\n", evaluate(sk(LinearRegression()), d), "\n")
    print("Ridge:\n", evaluate(sk(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))), d), "\n")
    print("Lasso:\n", evaluate(sk(make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=20000))), d), "\n")
    print("Random forest (300 trees):\n", evaluate(sk(RandomForestRegressor(300, random_state=1, n_jobs=-1)), d))
