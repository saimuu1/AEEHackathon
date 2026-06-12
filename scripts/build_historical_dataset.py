import asyncio
import csv
import io
import os
import sys
import time
import zipfile
from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Config ──────────────────────────────────────────────────────────
from backend.data.sites_data import SITES as DATA_SITES

SITES = {}
for s in DATA_SITES:
    hub = s["gas_hub"].lower().replace(" ", "_").replace("&", "")
    if hub == "katy": hub = "henry_hub" # proxy Katy to Henry Hub as fallback if missing
    SITES[s["id"]] = {
        "zone": s["settlement_point"],
        "gas_hub": hub,
        "lat": s["lat"],
        "lng": s["lng"]
    }

HEAT_RATE = 7.5
O_AND_M = 3.50
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "backend", "data", "historical_spreads.parquet")

EIA_API_KEY = os.getenv("EIA_API_KEY", "")  # set in .env; falls back to synthetic gas prices if absent


async def fetch_ercot_lmp_async(zones, days_back=90):
    """Fetch ERCOT DAM SPP prices (yearly bulk) for high-speed historical load."""
    print(f"📡 Fetching ERCOT DAM data for {zones} (yearly bulk)...")
    import gridstatus
    
    def _fetch():
        ercot = gridstatus.Ercot()
        end = pd.Timestamp.now(tz="US/Central")
        start = end - pd.Timedelta(days=days_back)
        
        years = list(set([start.year, end.year]))
        dfs = []
        for y in years:
            try:
                dfs.append(ercot.get_dam_spp(y))
            except Exception as e:
                print(f"   ⚠️ ERCOT year {y} error: {e}")
                
        if not dfs:
            return pd.DataFrame()
            
        df = pd.concat(dfs, ignore_index=True)
        # Filter to requested date window
        df = df[(df["Interval Start"] >= start) & (df["Interval Start"] <= end)].copy()
        
        df = df[df["Location"].isin(zones)].copy()
        
        # DAM is already hourly, but we'll floor and rename to match schema.
        # Make timestamp tz-naive to merge correctly with CAISO.
        df["hour"] = df["Interval Start"].dt.tz_convert("US/Central").dt.tz_localize(None).dt.floor("h")
        hourly = df.groupby(["Location", "hour"]).agg({"SPP": "mean"}).reset_index()
        hourly.rename(columns={"hour": "timestamp", "Location": "zone", "SPP": "LMP"}, inplace=True)
        return hourly

    return await asyncio.to_thread(_fetch)


async def fetch_caiso_chunk(client, node_id, start_dt, end_dt, all_rows):
    """Fetch a block of CAISO data up to 31 days, with retry on 429."""
    start_str = start_dt.strftime("%Y%m%dT07:00-0000")
    end_str = end_dt.strftime("%Y%m%dT07:00-0000")
    url = "http://oasis.caiso.com/oasisapi/SingleZip"
    params = {
        "queryname": "PRC_LMP",
        "market_run_id": "DAM",
        "startdatetime": start_str,
        "enddatetime": end_str,
        "node": node_id,
        "resultformat": 6,
        "version": 1,
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = await client.get(url, params=params, timeout=60, follow_redirects=True)
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content))
                for name in z.namelist():
                    if "INVALID" not in name:
                        with z.open(name) as f:
                            content = f.read().decode("utf-8")
                            reader = csv.DictReader(content.strip().split("\n"))
                            for row in reader:
                                if row.get("LMP_TYPE") == "LMP":
                                    try:
                                        ts_str = row.get("INTERVALSTARTTIME_GMT", "")
                                        ts = pd.Timestamp(ts_str).tz_convert("US/Central").tz_localize(None)
                                        all_rows.append({
                                            "timestamp": ts.floor("h"),
                                            "zone": node_id,
                                            "LMP": float(row.get("MW", 0)),
                                        })
                                    except: pass
                return  # success
            elif r.status_code == 429:
                wait = 5 * (2 ** attempt)
                print(f"   ⏳ CAISO {node_id} chunk {start_dt.date()} rate-limited, retrying in {wait}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                print(f"   ⚠️ CAISO {node_id} chunk {start_dt.date()} returned HTTP {r.status_code}")
                return
        except Exception as e:
            print(f"   ⚠️ CAISO {node_id} chunk {start_dt.date()} error: {e}")
            return
    print(f"   ❌ CAISO {node_id} chunk {start_dt.date()} failed after {max_retries} retries")


async def fetch_caiso_lmp_async(node_id, days_back=90):
    """Fetch CAISO LMP for a node by parallelizing 30-day chunk downloads."""
    print(f"📡 Parallel Fetching CAISO LMP for {node_id} in 30-day blocks...")
    all_rows = []
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=days_back)
    
    # Slice the date range into 30-day chunks to respect OASIS limits
    chunks = []
    current = start
    while current < end:
        chunk_end = min(current + pd.Timedelta(days=30), end)
        chunks.append((current, chunk_end))
        current = chunk_end

    async with httpx.AsyncClient() as client:
        for c_start, c_end in chunks:
            await fetch_caiso_chunk(client, node_id, c_start, c_end, all_rows)
            await asyncio.sleep(3.0)  # Longer delay between chunks to respect OASIS limits

    if all_rows:
        df = pd.DataFrame(all_rows)
        # Drop duplicates in case chunk boundaries overlapped slightly
        df = df.groupby(["zone", "timestamp"]).agg({"LMP": "mean"}).reset_index()
        print(f"   ✅ CAISO {node_id}: {len(df)} hourly rows")
        return df
    return None


async def fetch_eia_gas_prices_async(days_back=100):
    """Fetch gas prices via EIA API (async)."""
    print("📡 Fetching EIA gas prices (async)...")
    url = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"
    params = {
        "api_key": EIA_API_KEY,
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": "RNGWHHD",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": days_back,
    }
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params, timeout=15, follow_redirects=True)
            data = r.json()
            records = data.get("response", {}).get("data", [])
            if records:
                rows = []
                for rec in records:
                    if rec.get("value"):
                        rows.append({"date": pd.Timestamp(rec["period"]), "henry_hub": float(rec["value"])})
                df = pd.DataFrame(rows)
                # Derivatives
                for hub, diff in [("waha", -0.80), ("socal_border", 0.40), ("socal_citygate", 1.20), ("pge_citygate", 1.50), ("kern_river", 0.60)]:
                    df[hub] = df["henry_hub"] + diff
                print(f"   ✅ EIA: {len(df)} price points")
                return df
    except Exception as e:
        print(f"   ⚠️ EIA error: {e}")
    # Fallback
    dates = pd.date_range(end=datetime.now(), periods=days_back, freq="D")
    return pd.DataFrame({"date": dates, "henry_hub": 3.20, "waha": 2.40, "socal_border": 3.60, "socal_citygate": 4.50, "pge_citygate": 4.80, "kern_river": 3.80})


async def fetch_weather_async(lat, lng, days_back=90):
    """Fetch historical weather via Open-Meteo (async)."""
    end = datetime.now()
    start = end - timedelta(days=days_back)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lng, "hourly": "temperature_2m,wind_speed_10m", "temperature_unit": "fahrenheit", "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d"), "timezone": "America/Chicago"}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params, timeout=15, follow_redirects=True)
            data = r.json()
            h = data.get("hourly", {})
            if h.get("time"):
                return pd.DataFrame({"timestamp": pd.to_datetime(h["time"]), "temp_f": h["temperature_2m"], "wind_speed": h["wind_speed_10m"]})
    except: pass
    return None


async def build_dataset_async(days_back=90):
    """Orchestrate high-speed parallel collection."""
    print("=" * 60)
    print(f"⚡ HIGH-SPEED ASYNC COLLECTION ({days_back} days)")
    print("=" * 60)
    start_time = time.time()

    # 1. Parallel collection of market data
    ercot_zones = list(set([s["zone"] for s in SITES.values() if s["zone"].startswith("LZ_")]))
    unique_caiso_nodes = set([s["zone"] for s in SITES.values() if not s["zone"].startswith("LZ_")])
    
    ercot_task = fetch_ercot_lmp_async(ercot_zones, days_back)
    gas_task = fetch_eia_gas_prices_async(days_back + 10)
    
    print("📡 Launching fetches... (ERCOT & Gas in parallel, CAISO sequentially to avoid limits)")
    base_results = await asyncio.gather(ercot_task, gas_task)
    
    ercot_df = base_results[0]
    gas_df = base_results[1]
    
    caiso_dfs = []
    for node in unique_caiso_nodes:
        caiso_df = await fetch_caiso_lmp_async(node, days_back)
        if caiso_df is not None:
            caiso_dfs.append(caiso_df)
        await asyncio.sleep(5.0)  # cooldown between different nodes
    
    if ercot_df is None or len(ercot_df) == 0:
        print("   ⚠️ ERCOT fetch empty, using fallback")
        # Generate dummy ERCOT to prevent failure
        hours = pd.date_range(end=datetime.now(), periods=days_back*24, freq="h")
        ercot_df = pd.DataFrame({"timestamp": hours, "zone": "LZ_WEST", "LMP": np.random.normal(30, 20, len(hours))})

    all_lmp = pd.concat([ercot_df] + caiso_dfs, ignore_index=True)
    all_lmp["timestamp"] = pd.to_datetime(all_lmp["timestamp"])
    gas_df["date"] = pd.to_datetime(gas_df["date"])

    # 2. Parallel collection of weather for all sites
    print(f"\n📡 Fetching weather for {len(SITES)} sites in parallel...")
    weather_tasks = [fetch_weather_async(s["lat"], s["lng"], days_back) for s in SITES.values()]
    weather_results = await asyncio.gather(*weather_tasks)
    weather_map = {list(SITES.keys())[i]: res for i, res in enumerate(weather_results)}

    # 3. Merging
    print("\n🔨 Merging data and engineering features...")
    all_site_rows = []
    gas_df["date_key"] = gas_df["date"].dt.date

    for site_id, site_info in SITES.items():
        zone = site_info["zone"]
        site_lmp = all_lmp[all_lmp["zone"] == zone].copy()
        
        if len(site_lmp) == 0:
            print(f"   ⚠️ {site_id} ({zone}): No market data found")
            continue
        
        weather = weather_map.get(site_id)
        site_lmp["date"] = site_lmp["timestamp"].dt.date
        
        for _, row in site_lmp.iterrows():
            ts = row["timestamp"]
            gas_match = gas_df[gas_df["date_key"] <= ts.date()]
            gas_price = gas_match.iloc[0][site_info["gas_hub"]] if len(gas_match) > 0 else 3.0
            
            gen_cost = gas_price * HEAT_RATE + O_AND_M
            lmp = row["LMP"]
            
            temp, wind = 75.0, 10.0
            if weather is not None:
                w_match = weather[weather["timestamp"].dt.floor("h") == ts.floor("h")]
                if len(w_match) > 0:
                    temp, wind = w_match.iloc[0]["temp_f"], w_match.iloc[0]["wind_speed"]

            all_site_rows.append({
                "ts": ts, "site_id": site_id, "zone": zone,
                "lmp": round(lmp, 2), "temp_f": round(temp, 1), "wind_speed": round(wind, 1),
                "gas_price": round(gas_price, 2), "gen_cost": round(gen_cost, 2), "spread": round(lmp - gen_cost, 2),
                "hour": ts.hour, "weekday": ts.weekday(), "month": ts.month
            })

    if not all_site_rows:
        print("   ❌ CRITICAL: No merged data. Build failed.")
        return None

    df = pd.DataFrame(all_site_rows)
    df = df.sort_values(["site_id", "ts"]).reset_index(drop=True)
    
    # Feature Engineering (Restoring 24h lags)
    print("   📈 Engineering 6h and 24h lag features...")
    for site in df["site_id"].unique():
        mask = df["site_id"] == site
        df.loc[mask, "lmp_6h_lag"] = df.loc[mask, "lmp"].shift(6)
        df.loc[mask, "lmp_24h_lag"] = df.loc[mask, "lmp"].shift(24)
        df.loc[mask, "lmp_trend_6h"] = df.loc[mask, "lmp"] - df.loc[mask, "lmp"].shift(6)
        df.loc[mask, "lmp_trend_24h"] = df.loc[mask, "lmp"] - df.loc[mask, "lmp"].shift(24)
    
    df = df.dropna().reset_index(drop=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n✅ COLLECTION COMPLETE IN {elapsed:.1f}s")
    print(f"   Rows: {len(df)} | Sites: {df['site_id'].nunique()}")
    print(f"   Saved to: {OUTPUT_PATH}")
    return df

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
    asyncio.run(build_dataset_async(days_back=90))
