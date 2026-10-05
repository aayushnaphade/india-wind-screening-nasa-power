"""Core wind resource screening and chronological ML benchmark.

Loads the hourly NASA POWER files in data/raw/, derives 100 m wind-power
density (WPD) with hourly air density, and benchmarks five predictors on a
2024-2025 / 2026 split.  Produces Figs. 2-4 and 8 and Table IV of the paper.
Outputs go to figures/wind_only/ and tables/wind_only/.
"""
from __future__ import annotations

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
STYLE = {"font.size": 6.5, "axes.titlesize": 6.8, "axes.labelsize": 6.5, "xtick.labelsize": 5.8,
         "ytick.labelsize": 5.8, "legend.fontsize": 5.5, "font.family": "serif",
         "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
         "axes.linewidth": 0.6}

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
FIG = ROOT / "figures" / "wind_only"
TAB = ROOT / "tables" / "wind_only"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)


def load() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in sorted(DATA.glob("*_nasa_power_hourly_2024_2026.csv"))],
                  ignore_index=True)
    d["time_utc"] = pd.to_datetime(d["time_utc"], utc=True)
    cols = ["WS50M", "WD50M", "T2M", "RH2M", "PS"]
    d[cols] = d[cols].replace(-999, np.nan)
    d["v100"] = d["WS50M"] * 2 ** 0.14
    d["rho"] = d["PS"] * 1000 / (287.05 * (d["T2M"] + 273.15))
    d["wpd"] = 0.5 * d["rho"] * d["v100"] ** 3
    return d.sort_values(["city", "time_utc"]).reset_index(drop=True)


def resource(d: pd.DataFrame) -> dict:
    w = d.dropna(subset=["wpd"])
    city = (w.groupby("city", as_index=False)
            .agg(lat=("latitude", "first"), lon=("longitude", "first"),
                 hours=("wpd", "size"), v100=("v100", "mean"), rho=("rho", "mean"),
                 wpd=("wpd", "mean"), wpd_p95=("wpd", lambda x: x.quantile(0.95)),
                 share_above_5ms=("v100", lambda x: (x >= 5).mean())))
    city.sort_values("wpd", ascending=False).to_csv(TAB / "wind_city_summary.csv", index=False)

    # Fig. 1: coverage map, wind records only.
    world = gpd.read_file(ROOT / "data" / "data_natural_earth_countries.geojson")
    india = world[world["ADM0_A3"].eq("IND") | world["NAME"].eq("India")]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    world.cx[66:100, 5:38].plot(ax=ax, color="#f4f4f4", edgecolor="#b0b0b0", linewidth=0.45, zorder=0)
    india.plot(ax=ax, color="#f3f1e8", edgecolor="#333333", linewidth=1.1, zorder=1)
    ax.scatter(city["lon"], city["lat"], s=26, color="#0072B2", edgecolor="white",
               linewidth=0.4, alpha=0.9, zorder=2)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#0072B2",
                              markersize=6, label=f"NASA POWER wind records ({len(city)} cities)")],
              loc="lower left", framealpha=0.95)
    ax.set(xlim=(66, 100), ylim=(5, 37), xlabel="Longitude (°E)", ylabel="Latitude (°N)",
           title="Cities included in the wind screening")
    ax.grid(alpha=0.22)
    fig.tight_layout(); fig.savefig(FIG / "fig1_coverage.png", dpi=240); plt.close(fig)

    # Fig. 3: top 12 cities by mean 100 m WPD.
    top = city.nlargest(12, "wpd").sort_values("wpd")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 2.1))
        bars = ax.barh(top["city"], top["wpd"], color="#0072B2", height=0.7)
        ax.bar_label(bars, fmt="%.0f", padding=1.5, fontsize=5.2)
        ax.set(xlabel="Mean 100 m wind-power density (W m$^{-2}$)", xlim=(0, top["wpd"].max() * 1.1))
        ax.grid(axis="x", alpha=0.2); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.3); fig.savefig(FIG / "fig3_wpd_top12.png", dpi=300); plt.close(fig)

    # Fig. 4: monthly national mean WPD with inter-city spread.
    w = w.assign(month=w["time_utc"].dt.month)
    cm = w.groupby(["city", "month"])["wpd"].mean().unstack()
    nat = cm.mean()
    q25, q75 = cm.quantile(0.25), cm.quantile(0.75)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 1.8))
        ax.fill_between(nat.index, q25, q75, color="#0072B2", alpha=0.18, lw=0, label="Inter-city IQR")
        ax.plot(nat.index, nat.values, marker="o", ms=3, lw=1.2, color="#0072B2", label="Mean of 60 cities")
        ax.set(xlabel="Month", ylabel="Mean 100 m WPD (W m$^{-2}$)", xticks=range(1, 13),
               xticklabels=list("JFMAMJJASOND"))
        ax.legend(frameon=False, loc="upper left"); ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.3); fig.savefig(FIG / "fig4_monthly_wpd.png", dpi=300); plt.close(fig)

    jja = nat.loc[[6, 7, 8]].mean(); ond = nat.loc[[10, 11, 12]].mean()
    return {
        "cities": int(d["city"].nunique()), "records": int(len(d)),
        "valid_records": int(len(w)),
        "mean_v100": float(w["v100"].mean()), "mean_wpd": float(w["wpd"].mean()),
        "mean_rho": float(w["rho"].mean()),
        "top_cities": top.sort_values("wpd", ascending=False)[["city", "wpd", "v100"]].head(5).round(2).to_dict("records"),
        "bottom_city": city.nsmallest(1, "wpd")[["city", "wpd"]].round(2).to_dict("records"),
        "cities_wpd_ge_200": int((city["wpd"] >= 200).sum()),
        "cities_wpd_lt_100": int((city["wpd"] < 100).sum()),
        "median_city_wpd": float(city["wpd"].median()),
        "monthly_mean_wpd": nat.round(1).to_dict(),
        "peak_month": int(nat.idxmax()), "low_month": int(nat.idxmin()),
        "jja_to_ond_ratio": float(jja / ond),
    }


def benchmark(d: pd.DataFrame) -> dict:
    d = d.copy()
    d["wd_sin"] = np.sin(np.deg2rad(d["WD50M"])); d["wd_cos"] = np.cos(np.deg2rad(d["WD50M"]))
    h, doy = d["time_utc"].dt.hour, d["time_utc"].dt.dayofyear
    d["hour_sin"], d["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    d["lag24"] = d.groupby("city")["wpd"].shift(24)
    d["lag168"] = d.groupby("city")["wpd"].shift(168)
    feats = ["wd_sin", "wd_cos", "T2M", "RH2M", "PS", "latitude", "longitude",
             "hour_sin", "hour_cos", "doy_sin", "doy_cos", "lag24", "lag168"]
    keep = d[feats + ["wpd", "time_utc", "city"]].dropna()
    train = keep[keep["time_utc"].dt.year <= 2025]
    test = keep[keep["time_utc"].dt.year == 2026]
    fit = train.sample(n=min(250_000, len(train)), random_state=42)
    Xtr, ytr = fit[feats].to_numpy(), np.log1p(fit["wpd"].to_numpy())
    Xte, yte = test[feats].to_numpy(), test["wpd"].to_numpy()
    models = {
        "Persistence (24 h)": None,
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Random forest": RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features=0.8,
                                               random_state=42, n_jobs=-1),
        "Extra trees": ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, max_features=0.9,
                                           random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                                              max_leaf_nodes=31, l2_regularization=1.0,
                                                              random_state=42),
    }
    rows, preds = [], {}
    # Log-space predictions are bounded by the training range before back-
    # transforming; otherwise the linear model extrapolates to absurd WPD.
    ylo, yhi = ytr.min(), ytr.max()
    for name, m in models.items():
        p = (test["lag24"].to_numpy() if m is None
             else np.expm1(np.clip(m.fit(Xtr, ytr).predict(Xte), ylo, yhi)))
        p = np.maximum(p, 0); preds[name] = p
        rmse = mean_squared_error(yte, p) ** 0.5
        rows.append({"method": name, "test_records": len(yte), "MAE": mean_absolute_error(yte, p),
                     "RMSE": rmse, "R2": r2_score(yte, p), "nRMSE": 100 * rmse / yte.mean()})
    res = pd.DataFrame(rows).sort_values("RMSE")
    res.to_csv(TAB / "ml_wpd_benchmark.csv", index=False)
    best = res.iloc[0]["method"]; bp = preds[best]
    ev = test[["city", "wpd"]].assign(pred=bp)
    cityres = pd.DataFrame([{"city": c, "MAE": mean_absolute_error(g["wpd"], g["pred"]),
                             "RMSE": mean_squared_error(g["wpd"], g["pred"]) ** 0.5,
                             "R2": r2_score(g["wpd"], g["pred"])} for c, g in ev.groupby("city")])
    cityres.to_csv(TAB / "ml_best_model_city_metrics.csv", index=False)

    # Held-out diagnostics (Fig. 8).
    short = {"Random forest": "RF", "HistGradientBoosting": "HGB", "Extra trees": "ET",
             "Persistence (24 h)": "Persist.", "Ridge": "Ridge"}
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 3, figsize=(3.5, 1.45), gridspec_kw={"width_ratios": [0.8, 1, 1]})
        order = res["method"].tolist()[::-1]; lk = res.set_index("method")
        ax[0].barh([short[m] for m in order], [lk.loc[m, "RMSE"] for m in order], color="#0072B2")
        ax[0].set(xlabel="RMSE (W m$^{-2}$)", title="(a) Test error"); ax[0].grid(axis="x", alpha=0.2)
        idx = np.random.default_rng(42).choice(len(yte), size=min(10000, len(yte)), replace=False)
        ax[1].scatter(yte[idx], bp[idx], s=1, alpha=0.2, color="#009E73", rasterized=True, lw=0)
        lim = np.nanpercentile(np.r_[yte[idx], bp[idx]], 99.5)
        ax[1].plot([0, lim], [0, lim], "k--", lw=0.7)
        ax[1].set(xlabel="Observed (W m$^{-2}$)", ylabel="Predicted (W m$^{-2}$)",
                  title=f"(b) {short[best]} calibration", xlim=(0, lim), ylim=(0, lim)); ax[1].grid(alpha=0.2)
        r = bp - yte; lo, hi = np.percentile(r, [0.5, 99.5])
        ax[2].hist(r[(r >= lo) & (r <= hi)], bins=40, color="#E69F00", edgecolor="white", lw=0.2)
        ax[2].axvline(0, color="k", ls="--", lw=0.7)
        ax[2].set(xlabel="Error (W m$^{-2}$)", ylabel="Hours (×10$^3$)", title="(c) Residuals", xlim=(lo, hi))
        ax[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
        ax[2].grid(axis="y", alpha=0.2)
        fig.tight_layout(pad=0.3, w_pad=0.4)
        fig.savefig(FIG / "fig5_ml_diagnostics.png", dpi=300); plt.close(fig)

    # Fig. 2: workflow of the whole study (resource branch and ML branch).
    from matplotlib.patches import FancyBboxPatch
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(3.5, 1.9))
        ax.set(xlim=(0, 10), ylim=(0, 6.6)); ax.axis("off")
        boxes = {  # name: (x0, y0, w, h, colour, text)
            "data": (0.05, 5.0, 3.0, 1.45, "#E8F1F8", "NASA POWER (MERRA-2)\n60 cities, hourly\nJan 2024 – Sep 2026"),
            "qc": (3.5, 5.0, 3.0, 1.45, "#E8F1F8", "Audit and QC\n−999 hours removed\nshared grid cells flagged"),
            "phys": (6.95, 5.0, 3.0, 1.45, "#E8F1F8", "Physics\nρ from p, T;  v$_{100}$ (α = 0.14)\nhourly WPD"),
            "res": (0.05, 2.45, 4.6, 1.6, "#EAF5EE", "Resource screening\nWeibull (MLE) · CF (IEC density)\n"
                                                     "seasonal / diurnal · direction roses\nwind droughts · aggregation"),
            "ml": (5.35, 2.45, 4.6, 1.6, "#FCF1E3", "ML benchmark\nfeatures: direction, T, RH, p,\n"
                                                   "calendar, 24 h / 168 h lags\n5 models incl. persistence"),
            "out": (0.05, 0.1, 4.6, 1.45, "#EAF5EE", "Outputs\ncity / regional rankings\nFigs. 1, 3–7;  Tables II–III"),
            "eval": (5.35, 0.1, 4.6, 1.45, "#FCF1E3", "Chronological test\ntrain 2024–25 → test 2026\n"
                                                     "MAE, RMSE, R², SS, importance"),
        }
        edge = {"#E8F1F8": "#1F4E79", "#EAF5EE": "#1B6B3A", "#FCF1E3": "#9A5B00"}
        for x0, y0, w, h, col, text in boxes.values():
            ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                        fc=col, ec=edge[col], lw=0.7))
            ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=5.4, linespacing=1.25)
        arrow = dict(arrowstyle="-|>", color="#444444", lw=0.7, mutation_scale=6, shrinkA=0, shrinkB=0)

        def link(p, q):
            ax.annotate("", xy=q, xytext=p, arrowprops=arrow)

        gap = 0.06
        link((3.05 + gap, 5.725), (3.5 - gap, 5.725))
        link((6.5 + gap, 5.725), (6.95 - gap, 5.725))
        bus = 4.55  # branch line between the rows, clear of every box
        ax.plot([8.45, 8.45], [5.0 - gap, bus], color="#444444", lw=0.7)
        ax.plot([2.35, 8.45], [bus, bus], color="#444444", lw=0.7)
        link((2.35, bus), (2.35, 4.05 + gap))
        link((7.65, bus), (7.65, 4.05 + gap))
        link((2.35, 2.45 - gap), (2.35, 1.55 + gap))
        link((7.65, 2.45 - gap), (7.65, 1.55 + gap))
        fig.tight_layout(pad=0.1); fig.savefig(FIG / "fig2_workflow.png", dpi=300); plt.close(fig)

    return {"train_rows": len(fit), "test_rows": len(test), "best": best,
            "table": res.round(3).to_dict("records"),
            "city_r2_median": float(cityres["R2"].median()),
            "city_r2_min": float(cityres["R2"].min()), "city_r2_max": float(cityres["R2"].max()),
            "cities_r2_below_0": int((cityres["R2"] < 0).sum())}


if __name__ == "__main__":
    data = load()
    out = {"resource": resource(data), "ml": benchmark(data)}
    (TAB / "summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
