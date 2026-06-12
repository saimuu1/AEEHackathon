#!/usr/bin/env /opt/anaconda3/bin/python
"""Fast dataset build: ERCOT SPP + CAISO + EIA + Weather → parquet"""
import csv
import io
import os
import warnings
import zipfile
from datetime import datetime, timedelta

import gridstatus
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()  # read EIA_API_KEY etc. from .env

EIA_API_KEY = os.getenv("EIA_API_KEY", "")  # never hardcode secrets

print("=" * 60)
print("FAST Dataset Build")
print("=" * 60)

# ── 1. ERCOT SPP (use get_spp with DAM market) ──
print("\n📡 ERCOT Day-Ahead SPP...")
ercot = gridstatus.Ercot()
end_ts = pd.Timestamp.now(tz="US/Central")
start_ts = end_ts - pd.Timedelta(days=14)
df = ercot.get_spp(start_ts, end=end_ts, market="DAY_AHEAD_HOURLY", verbose=False)
zones = ["LZ_WEST", "LZ_NORTH", "LZ_HOUSTON", "LZ_SOUTH"]
df = df[df["Location"].isin(zones)].copy()
df.rename(columns={"Interval Start": "timestamp", "Location": "zone", "SPP": "lmp"}, inplace=True)
df = df[["timestamp", "zone", "lmp"]]
df["timestamp"] = df["timestamp"].dt.tz_localize(None)
print(f"   ✅ ERCOT: {len(df)} rows")
print(df.groupby("zone")["lmp"].agg(["mean", "min", "max"]).round(2))

# ── 2. CAISO Palo Verde ──
print("\n📡 CAISO Palo Verde...")
caiso_rows = []
for d in range(14):
    day = datetime.utcnow() - timedelta(days=d + 1)
    s = day.strftime("%Y%m%dT07:00-0000")
    e = (day + timedelta(days=1)).strftime("%Y%m%dT07:00-0000")
    try:
        r = requests.get("http://oasis.caiso.com/oasisapi/SingleZip", params={
            "queryname": "PRC_LMP", "market_run_id": "DAM",
            "startdatetime": s, "enddatetime": e,
            "node": "PALOVRDE_ASR-APND", "resultformat": 6, "version": 1
        }, timeout=30)
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            for name in z.namelist():
                if "INVALID" not in name:
                    with z.open(name) as f:
                        reader = csv.DictReader(f.read().decode("utf-8").strip().split("\n"))
                        for row in reader:
                            if row.get("LMP_TYPE") == "LMP":
                                try:
                                    ts = pd.Timestamp(row["INTERVALSTARTTIME_GMT"]).tz_convert("US/Central").tz_localize(None)
                                    caiso_rows.append({"timestamp": ts, "zone": "PALOVRDE", "lmp": float(row["MW"])})
                                except:
                                    pass
    except:
        pass

if caiso_rows:
    caiso = pd.DataFrame(caiso_rows)
    print(f"   ✅ CAISO: {len(caiso)} rows, avg=${caiso['lmp'].mean():.1f}/MWh")
    df = pd.concat([df, caiso], ignore_index=True)
else:
    print("   ⚠️  CAISO: synthetic fallback")
    hrs = pd.date_range(datetime.now() - timedelta(days=7), periods=168, freq="h")
    df = pd.concat([df, pd.DataFrame({"timestamp": hrs, "zone": "PALOVRDE",
                                       "lmp": np.random.normal(30, 12, 168).clip(5, 150)})], ignore_index=True)

# ── 3. EIA Gas Prices ──
print("\n📡 EIA Henry Hub...")
r = requests.get("https://api.eia.gov/v2/natural-gas/pri/fut/data/", params={
    "api_key": EIA_API_KEY,
    "frequency": "daily", "data[0]": "value",
    "facets[series][]": "RNGWHHD",
    "sort[0][column]": "period", "sort[0][direction]": "desc",
    "offset": 0, "length": 30
}, timeout=15)
gdata = r.json().get("response", {}).get("data", [])
hh = float([g for g in gdata if g.get("value")][0]["value"])
waha = hh - 0.80
print(f"   ✅ EIA: Henry Hub=${hh:.2f}, Waha=${waha:.2f}")

# ── 4. Weather (all 7 sites) ──
print("\n📡 Open-Meteo weather...")
sites = {
    "midland":     {"zone": "LZ_WEST",    "hub": "waha",  "lat": 31.99, "lng": -102.08},
    "odessa":      {"zone": "LZ_WEST",    "hub": "waha",  "lat": 31.85, "lng": -102.37},
    "abilene":     {"zone": "LZ_WEST",    "hub": "waha",  "lat": 32.45, "lng": -99.73},
    "houston":     {"zone": "LZ_HOUSTON", "hub": "hh",    "lat": 29.76, "lng": -95.37},
    "dallas":      {"zone": "LZ_NORTH",   "hub": "hh",    "lat": 32.78, "lng": -96.80},
    "san_antonio": {"zone": "LZ_SOUTH",   "hub": "hh",    "lat": 29.42, "lng": -98.49},
    "tucson":      {"zone": "PALOVRDE",   "hub": "socal", "lat": 32.22, "lng": -110.97},
}
wcache = {}
for sid, s in sites.items():
    try:
        wr = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": s["lat"], "longitude": s["lng"],
            "hourly": "temperature_2m,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "start_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "timezone": "America/Chicago",
        }, timeout=10).json()
        wcache[sid] = pd.DataFrame({
            "timestamp": pd.to_datetime(wr["hourly"]["time"]),
            "temp_f": wr["hourly"]["temperature_2m"],
            "wind_speed": wr["hourly"]["wind_speed_10m"],
        })
    except:
        wcache[sid] = None
print(f"   ✅ Weather: {sum(1 for v in wcache.values() if v is not None)}/{len(sites)} sites")

# ── 5. Merge into per-site dataset ──
print("\n📦 Merging dataset...")
hub_map = {"waha": waha, "hh": hh, "socal": hh + 0.40}
rows = []
for sid, s in sites.items():
    zlmp = df[df["zone"] == s["zone"]].sort_values("timestamp")
    gas = hub_map[s["hub"]]
    w = wcache.get(sid)
    for _, row in zlmp.iterrows():
        ts = row["timestamp"]
        lmp = row["lmp"]
        gc = gas * 7.5 + 3.50
        temp, wind = 75.0, 10.0
        if w is not None:
            wm = w[w["timestamp"].dt.floor("h") == pd.Timestamp(ts).floor("h")]
            if len(wm) > 0:
                temp = float(wm.iloc[0]["temp_f"] or 75)
                wind = float(wm.iloc[0]["wind_speed"] or 10)
        rows.append({
            "ts": ts, "site_id": sid, "zone": s["zone"], "lmp": round(lmp, 2),
            "temp_f": round(temp, 1), "wind_speed": round(wind, 1),
            "hour": ts.hour, "weekday": ts.weekday(), "month": ts.month,
            "gas_price": round(gas, 2), "gen_cost": round(gc, 2),
            "spread": round(lmp - gc, 2),
        })

result = pd.DataFrame(rows).sort_values(["site_id", "ts"]).reset_index(drop=True)

# Add lag features
for site in result["site_id"].unique():
    m = result["site_id"] == site
    result.loc[m, "lmp_6h_lag"] = result.loc[m, "lmp"].shift(6)
    result.loc[m, "lmp_24h_lag"] = result.loc[m, "lmp"].shift(24)
    result.loc[m, "lmp_trend_6h"] = result.loc[m, "lmp"] - result.loc[m, "lmp"].shift(6)
    result.loc[m, "lmp_trend_24h"] = result.loc[m, "lmp"] - result.loc[m, "lmp"].shift(24)
result = result.dropna().reset_index(drop=True)

# Save
out = "backend/data/historical_spreads.parquet"
result.to_parquet(out, index=False)
print(f"\n{'=' * 60}")
print(f"✅ SAVED: {out}")
print(f"   Rows: {len(result)}  |  Sites: {result['site_id'].nunique()}")
print(f"   Range: {result['ts'].min()} → {result['ts'].max()}")
for s2 in sorted(result["site_id"].unique()):
    d2 = result[result["site_id"] == s2]
    print(f"   {s2:15s}  avg=${d2['spread'].mean():+6.1f}  min=${d2['spread'].min():+7.1f}  max=${d2['spread'].max():+7.1f}")
print(f"{'=' * 60}")
