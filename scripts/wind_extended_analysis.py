"""Extended outcomes: Figs. 1, 5-7, 9, Tables II-III and robustness checks.

Builds on wind_only_analysis.py (same loading, same 100 m WPD target) and adds:
  * MLE Weibull fits per city (cf. Patidar et al., 2025)
  * capacity factor from a generic, density-corrected turbine curve
  * diurnal x seasonal structure in local time and by region
  * energy-weighted direction roses
  * wind droughts (20th-percentile definition of Qu et al., 2025) and the
    effect of spatial aggregation (cf. Dijkstra et al., 2025)
  * state-level comparison with Kumar et al. (2026)
  * shear-exponent sensitivity and inter-annual variability
  * grouped permutation importance and per-city skill of the RF benchmark
Outputs go to figures/wind_extended/ and tables/wind_extended/.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from wind_only_analysis import load

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures" / "wind_extended"
TAB = ROOT / "tables" / "wind_extended"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 6.8, "axes.labelsize": 6.5,
                     "xtick.labelsize": 5.8, "ytick.labelsize": 5.8, "legend.fontsize": 5.5,
                     "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "axes.linewidth": 0.6})
BLUE, GREEN, ORANGE, RED, PURPLE = "#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"

STATE = {
    "Gujarat": ["Ahmedabad", "Bhavnagar", "Bhuj", "Gandhinagar", "Jamnagar", "Rajkot", "Surat", "Vadodara"],
    "Rajasthan": ["Barmer", "Bikaner", "Jaipur", "Jaisalmer", "Jodhpur", "Kota", "Udaipur"],
    "Maharashtra": ["Aurangabad", "Mumbai", "Nagpur", "Nashik", "Pune", "Solapur"],
    "Karnataka": ["Belgaum", "Bengaluru", "Hubli Dharwad", "Mangalore", "Mysore"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli", "Tirunelveli", "Tuticorin"],
    "Andhra Pradesh": ["Kurnool", "Tirupati", "Vijayawada", "Visakhapatnam"],
    "Telangana": ["Hyderabad"], "Kerala": ["Kochi", "Thiruvananthapuram"], "Goa": ["Panaji"],
    "Madhya Pradesh": ["Bhopal", "Gwalior", "Indore", "Jabalpur"],
    "Uttar Pradesh": ["Agra", "Kanpur", "Lucknow", "Prayagraj", "Varanasi"],
    "North (other)": ["Chandigarh", "Gurugram", "New Delhi", "Ludhiana", "Dehradun", "Shimla"],
    "East": ["Bhubaneswar", "Kolkata", "Patna", "Ranchi"],
}
REGION_OF_STATE = {"Gujarat": "North-west", "Rajasthan": "North-west",
                   "Maharashtra": "Central", "Madhya Pradesh": "Central", "Goa": "Central",
                   "Karnataka": "South", "Tamil Nadu": "South", "Andhra Pradesh": "South",
                   "Telangana": "South", "Kerala": "South",
                   "Uttar Pradesh": "North", "North (other)": "North", "East": "East"}
CITY_STATE = {c: s for s, cs in STATE.items() for c in cs}
REGIONS = ["North-west", "Central", "South", "North", "East"]
RCOL = dict(zip(REGIONS, [BLUE, GREEN, ORANGE, PURPLE, RED]))

# Generic utility-scale curve: cut-in 3, rated 12, cut-out 25 m/s, cubic ramp.
VCI, VR, VCO = 3.0, 12.0, 25.0


def power_curve(v: np.ndarray) -> np.ndarray:
    p = np.clip((v ** 3 - VCI ** 3) / (VR ** 3 - VCI ** 3), 0, 1)
    return np.where((v < VCI) | (v >= VCO), 0.0, p)


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    d = d.dropna(subset=["wpd"]).copy()
    d["state"] = d["city"].map(CITY_STATE)
    d["region"] = d["state"].map(REGION_OF_STATE)
    assert d["region"].notna().all(), d.loc[d["region"].isna(), "city"].unique()
    # IEC 61400-12-1 density normalisation of speed before applying the curve.
    d["cf"] = power_curve(d["v100"] * (d["rho"] / 1.225) ** (1 / 3))
    local = d["time_utc"] + pd.Timedelta(hours=5, minutes=30)
    d["lhour"], d["month"], d["year"] = local.dt.hour, local.dt.month, local.dt.year
    d["date"] = local.dt.floor("D").dt.tz_localize(None)
    return d


# ---------------------------------------------------------------- city metrics
def city_metrics(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in d.groupby("city"):
        v = g["v100"].to_numpy()
        k, _, lam = stats.weibull_min.fit(v[v > 0.1], floc=0)
        calm = (v < VCI).astype(int)
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, calm, 0])))[::2]
        rows.append({"city": c, "state": g["state"].iat[0], "region": g["region"].iat[0],
                     "lat": g["latitude"].iat[0], "lon": g["longitude"].iat[0],
                     "v100": v.mean(), "wpd": g["wpd"].mean(), "rho": g["rho"].mean(),
                     "k": k, "c": lam,
                     "wpd_weibull": 0.5 * g["rho"].mean() * lam ** 3 * gamma(1 + 3 / k),
                     "cf": g["cf"].mean(), "below_cutin": calm.mean(),
                     "longest_calm_h": int(runs.max()) if len(runs) else 0})
    m = pd.DataFrame(rows).sort_values("wpd", ascending=False)
    m.to_csv(TAB / "city_weibull_cf.csv", index=False)
    return m


def fig_cf_map(m: pd.DataFrame) -> None:
    world = gpd.read_file(ROOT / "data" / "data_natural_earth_countries.geojson")
    india = world[world["ADM0_A3"].eq("IND") | world["NAME"].eq("India")]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    world.cx[66:100, 5:38].plot(ax=ax, color="#f4f4f4", edgecolor="#b0b0b0", linewidth=0.4, zorder=0)
    india.plot(ax=ax, color="#f3f1e8", edgecolor="#333333", linewidth=0.8, zorder=1)
    sc = ax.scatter(m["lon"], m["lat"], c=100 * m["cf"], s=12 + 900 * m["cf"] ** 1.5, cmap="viridis",
                    edgecolor="k", linewidth=0.3, zorder=2)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Screening capacity factor (%)")
    ax.set(xlim=(67, 92), ylim=(7, 33), xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    ax.grid(alpha=0.2)
    fig.tight_layout(pad=0.3); fig.savefig(FIG / "fig6_cf_map.png", dpi=300); plt.close(fig)


# ------------------------------------------------------------ diurnal/seasonal
def fig_diurnal(d: pd.DataFrame) -> dict:
    cm = d.groupby(["city", "month", "lhour"])["wpd"].mean()
    heat = cm.groupby(["month", "lhour"]).mean().unstack()
    reg = (d.groupby(["region", "city", "lhour"])["wpd"].mean()
           .groupby(["region", "lhour"]).mean().unstack())
    regn = reg.div(reg.mean(axis=1), axis=0)
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.85), gridspec_kw={"width_ratios": [1.2, 1]})
    im = ax[0].imshow(heat.values, aspect="auto", origin="lower", cmap="magma_r",
                      extent=(-0.5, 23.5, 0.5, 12.5))
    ax[0].set(xlabel="Local hour (IST)", ylabel="Month", yticks=range(1, 13), xticks=range(0, 24, 6),
              title="(a) WPD by month and hour")
    fig.colorbar(im, ax=ax[0], pad=0.02).set_label("W m$^{-2}$")
    for r in REGIONS:
        ax[1].plot(regn.columns, regn.loc[r], color=RCOL[r], lw=1.4, label=r)
    ax[1].axhline(1, color="k", lw=0.6, ls=":")
    ax[1].set(xlabel="Local hour (IST)", ylabel="WPD / daily mean", xticks=range(0, 24, 6),
              title="(b) Diurnal cycle by region")
    ax[1].legend(ncol=2, frameon=False, fontsize=4.8, handlelength=1.2, columnspacing=0.6, loc="upper center"); ax[1].set_ylim(0.6, 1.55); ax[1].grid(alpha=0.2)
    fig.tight_layout(pad=0.3); fig.savefig(FIG / "fig7_diurnal.png", dpi=300); plt.close(fig)
    nat = heat.mean()
    return {"peak_hour": int(nat.idxmax()), "min_hour": int(nat.idxmin()),
            "peak_to_min": float(nat.max() / nat.min()),
            "region_amplitude": {r: round(float(regn.loc[r].max() - regn.loc[r].min()), 2) for r in REGIONS},
            "region_peak_hour": {r: int(regn.loc[r].idxmax()) for r in REGIONS},
            "july_peak": float(heat.loc[7].max()), "july_peak_hour": int(heat.loc[7].idxmax())}


# --------------------------------------------------------------- energy roses
def fig_roses(d: pd.DataFrame, cities=("Bhuj", "Tuticorin", "Bengaluru")) -> dict:
    edges = np.deg2rad(np.arange(-11.25, 360, 22.5))
    fig, ax = plt.subplots(1, 3, figsize=(3.5, 1.55), subplot_kw={"projection": "polar"})
    out = {}
    for a, c in zip(ax, cities):
        g = d[d["city"] == c]
        wd = np.deg2rad((g["WD50M"].to_numpy() + 11.25) % 360 - 11.25)
        e, _ = np.histogram(wd, bins=edges, weights=g["wpd"].to_numpy())
        h, _ = np.histogram(wd, bins=edges)
        e, h = 100 * e / e.sum(), 100 * h / h.sum()
        ctr = np.deg2rad(np.arange(0, 360, 22.5))
        a.bar(ctr, e, width=np.deg2rad(20), color=BLUE, alpha=0.85, label="Energy share")
        a.plot(np.r_[ctr, ctr[0]], np.r_[h, h[0]], color=ORANGE, lw=1.1, label="Time share")
        a.set_theta_zero_location("N"); a.set_theta_direction(-1)
        a.set_title(c, pad=6); a.tick_params(labelsize=4.5, pad=-2)
        a.set_xticks(np.deg2rad([0, 90, 180, 270])); a.set_xticklabels(["N", "E", "S", "W"])
        a.set_yticklabels([])
        sw = (ctr >= np.deg2rad(202.5)) & (ctr <= np.deg2rad(292.5))
        top = int(np.argmax(e))
        out[c] = {"sw_w_energy_share": float(e[sw].sum()), "sw_w_time_share": float(h[sw].sum()),
                  "top_sector_deg": float(np.rad2deg(ctr[top])), "top_sector_energy": float(e[top])}
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(pad=0.2, rect=(0, 0.08, 1, 1)); fig.savefig(FIG / "fig8_energy_roses.png", dpi=300); plt.close(fig)
    return out


# --------------------------------------------------------------- wind droughts
def runs_of(mask: np.ndarray) -> np.ndarray:
    x = np.diff(np.r_[0, mask.astype(int), 0])
    return np.flatnonzero(x == -1) - np.flatnonzero(x == 1)


def droughts(d: pd.DataFrame, m: pd.DataFrame) -> dict:
    # Gandhinagar duplicates Ahmedabad's grid cell, so it is left out of aggregates.
    dd = d[d["city"] != "Gandhinagar"]
    daily = dd.groupby(["city", "date"])["cf"].mean().unstack(0)
    daily = daily[daily.notna().all(axis=1)]
    region_of = {c: REGION_OF_STATE[CITY_STATE[c]] for c in daily.columns}
    reg_daily = daily.T.groupby(region_of).mean().T[REGIONS]
    nat_daily = daily.mean(axis=1)

    # Relative drought (Qu et al. 2025): day below the city's own 20th percentile.
    thr = daily.quantile(0.20)
    dmask = daily.lt(thr)
    city_rows = []
    for c in daily.columns:
        r = runs_of(dmask[c].to_numpy())
        city_rows.append({"city": c, "region": region_of[c], "events_ge3d": int((r >= 3).sum()),
                          "longest_days": int(r.max()), "p20_cf": float(thr[c])})
    cd = pd.DataFrame(city_rows); cd.to_csv(TAB / "city_droughts.csv", index=False)
    month = daily.index.month
    days_per_month = pd.Series(1, index=daily.index).groupby(month).sum()
    mfreq = dmask.groupby(month).sum().T.groupby(region_of).sum().T
    ncity = pd.Series(region_of).value_counts()
    mfreq = (mfreq.div(days_per_month, axis=0) / ncity)[REGIONS]

    # Absolute low-output days and the effect of aggregation.
    low = 0.05
    single = (daily < low).mean()
    nat_runs = runs_of((nat_daily < low).to_numpy())
    agg = {"median_city": float(single.median()),
           "best_city_share": float(single.min()),
           "region": {r: float((reg_daily[r] < low).mean()) for r in REGIONS},
           "national": float((nat_daily < low).mean()),
           "national_longest_days": int(nat_runs.max()) if len(nat_runs) else 0,
           "cv_city_median": float((daily.std() / daily.mean()).median()),
           "cv_national": float(nat_daily.std() / nat_daily.mean())}
    corr = reg_daily.corr()

    fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.3))
    for r in REGIONS:
        ax[0, 0].plot(mfreq.index, 100 * mfreq[r], marker="o", ms=2.5, color=RCOL[r], lw=1.2, label=r)
    ax[0, 0].axhline(20, color="k", ls=":", lw=0.7)
    ax[0, 0].set(xticks=range(1, 13, 2), xlabel="Month", ylabel="Drought days (%)",
                 title="(a) Drought days by month")
    ax[0, 0].set_ylim(0, 80)
    ax[0, 0].legend(ncol=3, frameon=False, loc="upper left", fontsize=4.6, handlelength=1.0, columnspacing=0.5); ax[0, 0].grid(alpha=0.2)
    allruns = np.concatenate([runs_of(dmask[c].to_numpy()) for c in daily.columns])
    bins = np.arange(1, allruns.max() + 2)
    ax[0, 1].hist(allruns, bins=bins, color=BLUE, edgecolor="white")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set(xlabel="Event duration (days)", ylabel="Events (all cities)",
                 title="(b) Drought-event durations")
    ax[0, 1].grid(alpha=0.2)
    im = ax[1, 0].imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax[1, 0].set(xticks=range(5), yticks=range(5), title="(c) Daily regional CF correlation")
    ab = ["NW", "C", "S", "N", "E"]
    ax[1, 0].set_xticklabels(ab); ax[1, 0].set_yticklabels(ab)
    for i in range(5):
        for j in range(5):
            ax[1, 0].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=4.8,
                          color="white" if abs(corr.values[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax[1, 0], shrink=0.85)
    q = np.linspace(0, 100, 201)
    ax[1, 1].plot(q, np.percentile(daily["Bhuj"], 100 - q) * 100, color=RED, label="Bhuj")
    ax[1, 1].plot(q, np.percentile(daily.median(axis=1), 100 - q) * 100, color="grey", ls="--",
                  label="Median city")
    ax[1, 1].plot(q, np.percentile(reg_daily["South"], 100 - q) * 100, color=ORANGE, label="South")
    ax[1, 1].plot(q, np.percentile(nat_daily, 100 - q) * 100, color=BLUE, lw=1.8, label="All-India (59)")
    ax[1, 1].axhline(100 * low, color="k", ls=":", lw=0.7)
    ax[1, 1].set(xlabel="Days exceeded (%)", ylabel="Daily CF (%)", title="(d) Daily CF duration curves")
    ax[1, 1].legend(frameon=False, fontsize=4.8); ax[1, 1].grid(alpha=0.2)
    fig.tight_layout(pad=0.3); fig.savefig(FIG / "fig9_droughts.png", dpi=300); plt.close(fig)

    nd = (month.isin([11, 12, 1, 2]))
    return {"days": int(len(daily)),
            "longest_city": cd.nlargest(3, "longest_days")[["city", "longest_days"]].to_dict("records"),
            "median_longest_days": float(cd["longest_days"].median()),
            "median_events_ge3d": float(cd["events_ge3d"].median()),
            "share_drought_days_NDJF": float(dmask[nd].sum().sum() / dmask.sum().sum()),
            "share_days_NDJF": float(nd.mean()),
            "monthly_drought_freq_all": (dmask.groupby(month).mean().mean(axis=1) * 100).round(1).to_dict(),
            "region_monthly_peak": {r: int(mfreq[r].idxmax()) for r in REGIONS},
            "aggregation": agg, "region_corr": corr.round(2).to_dict()}


# -------------------------------------------------------- state-level compare
def states(m: pd.DataFrame) -> pd.DataFrame:
    s = (m[m["city"] != "Gandhinagar"].groupby("state")
         .agg(cities=("city", "size"), wpd=("wpd", "mean"), cf=("cf", "mean"), k=("k", "mean"))
         .sort_values("wpd", ascending=False))
    s.to_csv(TAB / "state_summary.csv")
    return s


# ------------------------------------------------ shear sensitivity & years
def sensitivity(d: pd.DataFrame, m: pd.DataFrame) -> dict:
    alphas = [0.10, 0.14, 0.20, 0.25]
    v50 = d["WS50M"].to_numpy(); rho = d["rho"].to_numpy()
    res = []
    for a in alphas:
        v = v50 * 2 ** a
        res.append({"alpha": a, "wpd": float(np.mean(0.5 * rho * v ** 3)),
                    "cf": float(np.mean(power_curve(v * (rho / 1.225) ** (1 / 3))))})
    base_std = float(np.mean(0.5 * 1.225 * (v50 * 2 ** 0.14) ** 3))
    sres = pd.DataFrame(res)

    win = d[(d["month"] <= 8)]
    cy = win.groupby(["city", "year"])["wpd"].mean().unstack()
    nat = cy.mean()
    rho_rank = {f"{a}-{b}": float(stats.spearmanr(cy[a], cy[b])[0])
                for a, b in [(2024, 2025), (2024, 2026), (2025, 2026)]}
    iav = (cy.std(axis=1) / cy.mean(axis=1))

    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.75))
    sc = ax[0].scatter(m["wpd"], 100 * m["cf"], c=m["k"], cmap="viridis", s=6, edgecolor="k", linewidth=0.2)
    fig.colorbar(sc, ax=ax[0], pad=0.02).set_label("Weibull k")
    ax[0].set(xlabel="Mean 100 m WPD (W m$^{-2}$)", ylabel="Screening CF (%)",
              title="(a) CF against WPD")
    ax[0].grid(alpha=0.2)
    lim = [0, cy.values.max() * 1.05]
    ax[1].scatter(cy[2024], cy[2025], s=5, color=BLUE, label="2025")
    ax[1].scatter(cy[2024], cy[2026], s=5, color=RED, marker="^", label="2026")
    ax[1].plot(lim, lim, "k--", lw=0.8)
    ax[1].set(xlim=lim, ylim=lim, xlabel="Jan–Aug 2024 WPD (W m$^{-2}$)",
              ylabel="Jan–Aug WPD, later year", title="(b) Inter-annual consistency")
    ax[1].legend(frameon=False); ax[1].grid(alpha=0.2)
    fig.tight_layout(pad=0.3); fig.savefig(FIG / "fig10_sensitivity_years.png", dpi=300); plt.close(fig)
    return {"alpha": sres.round(4).to_dict("records"),
            "std_density_wpd": base_std,
            "national_janaug": nat.round(1).to_dict(),
            "spearman": rho_rank, "city_iav_median": float(iav.median()),
            "city_iav_max": {"city": iav.idxmax(), "cv": float(iav.max())}}


# ----------------------------------------------------------- ML attribution
def ml_attribution(d: pd.DataFrame, m: pd.DataFrame) -> dict:
    d = d.sort_values(["city", "time_utc"]).copy()
    d["wd_sin"] = np.sin(np.deg2rad(d["WD50M"])); d["wd_cos"] = np.cos(np.deg2rad(d["WD50M"]))
    h, doy = d["time_utc"].dt.hour, d["time_utc"].dt.dayofyear
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    d["lag24"] = d.groupby("city")["wpd"].shift(24)
    d["lag168"] = d.groupby("city")["wpd"].shift(168)
    groups = {"Wind direction": ["wd_sin", "wd_cos"], "Temperature": ["T2M"], "Humidity": ["RH2M"],
              "Pressure": ["PS"], "Location": ["latitude", "longitude"],
              "Hour of day": ["hour_sin", "hour_cos"], "Day of year": ["doy_sin", "doy_cos"],
              "Lag 24 h": ["lag24"], "Lag 168 h": ["lag168"]}
    feats = [f for g in groups.values() for f in g]
    keep = d[feats + ["wpd", "time_utc", "city"]].dropna()
    train = keep[keep["time_utc"].dt.year <= 2025].sample(n=250_000, random_state=42)
    test = keep[keep["time_utc"].dt.year == 2026]
    ytr = np.log1p(train["wpd"].to_numpy()); lo, hi = ytr.min(), ytr.max()
    rf = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features=0.8,
                               random_state=42, n_jobs=-1).fit(train[feats].to_numpy(), ytr)
    pred = lambda X: np.maximum(np.expm1(np.clip(rf.predict(X), lo, hi)), 0)

    s = test.sample(n=60_000, random_state=7)
    Xs, ys = s[feats].to_numpy(), s["wpd"].to_numpy()
    base = mean_squared_error(ys, pred(Xs)) ** 0.5
    rng = np.random.default_rng(0); imp = {}
    for g, cols in groups.items():
        deltas = []
        for _ in range(3):
            Xp = Xs.copy(); perm = rng.permutation(len(Xp))
            for c in cols:
                j = feats.index(c); Xp[:, j] = Xp[perm, j]
            deltas.append(mean_squared_error(ys, pred(Xp)) ** 0.5 - base)
        imp[g] = float(np.mean(deltas))
    imp = pd.Series(imp).sort_values()

    tp = pred(test[feats].to_numpy())
    ev = test[["city", "wpd"]].assign(p=tp, pers=test["lag24"].to_numpy())
    cr = ev.groupby("city").apply(lambda g: pd.Series({
        "R2": r2_score(g["wpd"], g["p"]),
        "skill": 1 - mean_squared_error(g["wpd"], g["p"]) ** 0.5
        / mean_squared_error(g["wpd"], g["pers"]) ** 0.5})).reset_index()
    cr = cr.merge(m[["city", "wpd", "region", "k"]], on="city")
    cr.to_csv(TAB / "ml_city_skill.csv", index=False)
    rr = stats.spearmanr(cr["wpd"], cr["R2"])[0]
    rk = stats.spearmanr(cr["k"], cr["R2"])[0]

    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.8), gridspec_kw={"width_ratios": [1, 1.1]})
    ax[0].barh(imp.index, imp.values, color=BLUE)
    ax[0].set(xlabel="ΔRMSE when permuted (W m$^{-2}$)", title="(a) Permutation importance")
    ax[0].grid(axis="x", alpha=0.2)
    for r in REGIONS:
        g = cr[cr["region"] == r]
        ax[1].scatter(g["wpd"], g["R2"], s=6, color=RCOL[r], label=r)
    ax[1].set(xlabel="City mean WPD (W m$^{-2}$)", ylabel="Per-city R$^2$ (2026)",
              title="(b) RF skill by city")
    ax[1].legend(frameon=False, loc="lower right", fontsize=4.6, handletextpad=0.1, borderaxespad=0.2); ax[1].grid(alpha=0.2)
    fig.tight_layout(pad=0.3); fig.savefig(FIG / "fig11_ml_attribution.png", dpi=300); plt.close(fig)
    return {"base_rmse_sample": float(base), "importance": imp.round(2).to_dict(),
            "spearman_r2_wpd": float(rr), "spearman_r2_k": float(rk),
            "cities_beating_persistence": int((cr["skill"] > 0).sum()),
            "skill_median": float(cr["skill"].median()),
            "region_r2_median": cr.groupby("region")["R2"].median().round(2).to_dict()}


if __name__ == "__main__":
    d = prepare(load())
    m = city_metrics(d)
    fig_cf_map(m)
    out = {
        "weibull": {"k_median": float(m["k"].median()), "k_min": m.nsmallest(1, "k")[["city", "k"]].round(2).to_dict("records"),
                    "k_max": m.nlargest(1, "k")[["city", "k"]].round(2).to_dict("records"),
                    "wpd_bias_pct_median": float((100 * (m["wpd_weibull"] / m["wpd"] - 1)).median()),
                    "wpd_bias_pct_range": [float((100 * (m["wpd_weibull"] / m["wpd"] - 1)).min()),
                                           float((100 * (m["wpd_weibull"] / m["wpd"] - 1)).max())],
                    "r_weibull_emp": float(np.corrcoef(m["wpd"], m["wpd_weibull"])[0, 1])},
        "cf": {"mean": float(m["cf"].mean()), "median": float(m["cf"].median()),
               "top": m.nlargest(5, "cf")[["city", "cf"]].round(3).to_dict("records"),
               "bottom": m.nsmallest(3, "cf")[["city", "cf"]].round(3).to_dict("records"),
               "ge_25pct": int((m["cf"] >= 0.25).sum()), "lt_10pct": int((m["cf"] < 0.10).sum()),
               "spearman_cf_wpd": float(stats.spearmanr(m["cf"], m["wpd"])[0]),
               "below_cutin_median": float(m["below_cutin"].median()),
               "longest_calm": m.nlargest(3, "longest_calm_h")[["city", "longest_calm_h"]].to_dict("records"),
               "monthly_cf": (d.groupby(["city", "month"])["cf"].mean().groupby("month").mean() * 100).round(1).to_dict()},
        "table3": m.head(10)[["city", "state", "v100", "k", "c", "wpd", "cf", "below_cutin", "longest_calm_h"]].round(3).to_dict("records"),
        "states": states(m).round(3).reset_index().to_dict("records"),
        "diurnal": fig_diurnal(d),
        "roses": fig_roses(d),
        "droughts": droughts(d, m),
        "sensitivity": sensitivity(d, m),
        "ml": ml_attribution(d, m),
    }
    (TAB / "summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
