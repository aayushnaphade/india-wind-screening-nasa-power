"""Download hourly NASA POWER data for the 60 study cities.

City coordinates are read from data/cities.csv.  Files are written to
data/raw/ and existing files are skipped, so the frozen snapshot shipped with
this repository is not overwritten.  Use --out to download a fresh copy
elsewhere and compare it with the snapshot (NASA may reprocess recent months).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://power.larc.nasa.gov/api/temporal/hourly/point"
START, END = "20240101", "20260913"
PARAMETERS = "ALLSKY_SFC_SW_DWN,WS50M,WD50M,T2M,RH2M,PS"


def fetch(city: str, lat: float, lon: float) -> pd.DataFrame:
    params = {"parameters": PARAMETERS, "community": "RE", "longitude": f"{lon:.4f}",
              "latitude": f"{lat:.4f}", "start": START, "end": END, "format": "JSON",
              "time-standard": "UTC"}
    for attempt in range(4):
        r = requests.get(API, params=params, timeout=180)
        if r.ok:
            frame = pd.DataFrame(r.json()["properties"]["parameter"])
            frame.index = pd.to_datetime(frame.index, format="%Y%m%d%H", utc=True)
            frame.index.name = "time_utc"
            return frame.reset_index().assign(city=city, latitude=lat, longitude=lon)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * (attempt + 1))
            continue
        raise RuntimeError(f"{city}: {r.status_code} {r.text[:300]}")
    raise RuntimeError(f"{city}: request failed after retries")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    cities = pd.read_csv(ROOT / "data" / "cities.csv")
    for i, c in enumerate(cities.itertuples(), start=1):
        target = out / f"{c.city.lower().replace(' ', '_')}_nasa_power_hourly_2024_2026.csv"
        if target.exists():
            print(f"[{i}/{len(cities)}] exists: {c.city}")
            continue
        print(f"[{i}/{len(cities)}] downloading: {c.city}", flush=True)
        fetch(c.city, c.latitude, c.longitude).to_csv(target, index=False)
        time.sleep(0.25)
    files = sorted(out.glob("*_nasa_power_hourly_2024_2026.csv"))
    summary = {"source": "NASA POWER hourly API", "parameters": PARAMETERS.split(","), "start": START,
               "end": END, "files": len(files),
               "records": int(sum(sum(1 for _ in open(f, encoding="utf-8")) - 1 for f in files))}
    (out / "download_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
