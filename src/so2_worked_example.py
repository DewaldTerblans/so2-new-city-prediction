"""
STTN327 SO2 project - Worked example: predict one city that was left out of the fit.
Fits the mixed-effects model on the other 40 cities, then predicts the held-out city
  k = 0 : no SO2 readings from the city   -> Index, with a 90% conformal band
  k = 1 : first quarter revealed          -> Index + lambda * (average miss)
and draws the picture used in the report and the video.
Run:  python so2_worked_example.py [path/to/quarterly_SO2_data.csv] [city]
Needs: pandas numpy statsmodels matplotlib
Writes: worked_example_<city>.png
"""
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "quarterly_SO2_data/quarterly_SO2_data.csv"
CITY = sys.argv[2] if len(sys.argv) > 2 else "San Francisco"

d = pd.read_csv(CSV)
d["quarter"] = d["quarter"].astype(int)
d = d.sort_values(["city", "year", "quarter"]).reset_index(drop=True)
d["t"] = d.groupby("city").cumcount()
for q in (2, 3, 4):
    d[f"Q{q}"] = (d.quarter == q).astype(float)
XCOLS = ["temperature", "windspeed", "precipitation", "precipitationdays",
         "manufacturing_enterprises", "population_thousands", "Q2", "Q3", "Q4"]
FORMULA = "SO2 ~ " + " + ".join(XCOLS)


def fit(data):
    return smf.mixedlm(FORMULA, data, groups=data["city"]).fit(reml=True)


def index_of(m, data):
    b = m.fe_params
    return b["Intercept"] + data[XCOLS].values @ b[XCOLS].values


# ---- 90% conformal band (same rule as so2_mixed_effects.py): worst miss per city, 38th smallest of 41
worst, errs = {}, {}
for c in d.city.unique():
    tr, te = d[d.city != c], d[d.city == c]
    m = fit(tr)
    idx = index_of(m, te)
    worst[c] = np.abs(te.SO2.values - idx).max()
    errs[c] = np.sqrt(np.mean((te.SO2.values - idx) ** 2))
BAND = np.sort(list(worst.values()))[int(np.ceil((len(worst) + 1) * 0.90)) - 1]
median_err = float(np.median(list(errs.values())))

# ---- the held-out city
tr, te = d[d.city != CITY], d[d.city == CITY].sort_values("t")
m = fit(tr)
tau2, sigma2 = float(m.cov_re.iloc[0, 0]), float(m.scale)
idx = index_of(m, te)
lam = tau2 / (tau2 + sigma2)                       # k = 1
pred1 = idx + lam * (te.SO2.values[0] - idx[0])
actual = te.SO2.values
rmse0 = np.sqrt(np.mean((actual - idx) ** 2))
rmse1 = np.sqrt(np.mean((actual[1:] - pred1[1:]) ** 2))
inside = np.mean(np.abs(actual - idx) <= BAND)
print(f"{CITY}: k=0 RMSE {rmse0:.1f} (median city {median_err:.1f}); "
      f"k=1 RMSE on remaining quarters {rmse1:.1f}; band +/-{BAND:.2f}; "
      f"{inside:.0%} of quarters inside the band; lambda = {lam:.3f}")

# ---- figure
BLUE, ORANGE, INK, GREY = "#2A78D6", "#EB6834", "#0B0B0B", "#6B6A66"
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
x = np.arange(12)
fig, ax = plt.subplots(figsize=(9.4, 5.1))
ax.fill_between(x, np.maximum(idx - BAND, 0), idx + BAND, color=BLUE, alpha=0.10, linewidth=0)
ax.plot(x, idx, color=BLUE, lw=2.6, marker="o", ms=5, zorder=3)
ax.plot(x[1:], pred1[1:], color=ORANGE, lw=2.6, marker="o", ms=5, zorder=4)
ax.scatter(x, actual, s=70, color=INK, zorder=5)
# the quarter we are allowed to see
ax.scatter([0], [actual[0]], s=260, facecolor="none", edgecolor=ORANGE, linewidth=2.2, zorder=6)
ax.text(0, actual[0] - 4.0, "first\nquarter\nseen", color=ORANGE, fontsize=11, ha="center", va="top", linespacing=1.0)
# direct labels instead of a legend
ax.text(11.25, idx[-1] + 1.5, "No city data\n(k = 0)", color=BLUE, fontsize=13, fontweight="bold", va="center")
ax.text(11.25, pred1[-1] - 0.5, "After one quarter\n(k = 1)", color=ORANGE, fontsize=13, fontweight="bold", va="center")
ax.text(11.25, actual[-1] - 7.0, "● Actual SO2", color=INK, fontsize=13, fontweight="bold", va="center")
ax.text(5.5, idx[4] + BAND - 4.0, "90% band for a city with no data", color=BLUE, fontsize=12, ha="center", alpha=0.9)
labels = [f"Q{q}" for q in te.quarter]
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
for i, yr in enumerate(sorted(te.year.unique())):
    ax.text(i * 4 + 1.5, -0.12, str(int(yr)), transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=13, color=GREY, fontweight="bold")
for b in (3.5, 7.5):
    ax.axvline(b, color="#DDDDD8", lw=1, zorder=0)
ax.set_xlim(-0.6, 13.4)
ax.set_ylim(0, None)
ax.set_ylabel("SO2 (µg/m³)")
ax.set_title(f"{CITY}, left out of the fit\n", loc="left", fontsize=16, fontweight="bold", color=INK, pad=22)
ax.text(0, 1.045, f"Typical error: {rmse0:.1f} µg/m³ with no city data  →  {rmse1:.1f} after one quarter",
        transform=ax.transAxes, fontsize=13, color=GREY, va="bottom")
fig.tight_layout()
out = f"worked_example_{CITY.replace(' ', '_')}.png"
fig.savefig(out, dpi=200)
print("saved", out)
