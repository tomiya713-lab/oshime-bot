# -*- coding: utf-8 -*-
"""
Weekly job: Nikkei225 beta metrics + J-Quants sector labels.

Outputs (CSV) are saved under: reports/beta/
- beta_by_stock.csv (latest overwrite)
- beta_by_sector.csv (latest overwrite)
- beta_by_stock_YYYYMMDD.csv (date-stamped archive)
- beta_by_sector_YYYYMMDD.csv (date-stamped archive)

Intended to be run from GitHub Actions weekly.
Another script can read the latest overwrite files.
"""

import os
import re
import time
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# -------------------------
# 0) Universe: Nikkei225 (.T tickers)
# -------------------------
NIKKEI225_TICKERS: List[str] = [
    '1332.T','1605.T','1721.T','1801.T','1802.T','1803.T','1808.T','1812.T',
    '1925.T','1928.T','1963.T','2002.T','2269.T','2282.T','2413.T','2432.T',
    '2501.T','2502.T','2503.T','2768.T','2801.T','2802.T','2871.T','2914.T',
    '3086.T','3092.T','3099.T','3289.T','3382.T','3401.T','3402.T','3405.T',
    '3407.T','3436.T','3659.T','3861.T','4004.T','4005.T','4021.T','4042.T',
    '4043.T','4061.T','4063.T','4151.T','4183.T','4188.T','4208.T','4307.T',
    '4324.T','4385.T','4452.T','4502.T','4503.T','4506.T','4507.T','4519.T',
    '4523.T','4543.T','4568.T','4578.T','4661.T','4689.T','4704.T','4751.T',
    '4755.T','4901.T','4902.T','4911.T','5019.T','5020.T','5101.T','5108.T',
    '5201.T','5214.T','5233.T','5301.T','5332.T','5333.T','5401.T','5406.T',
    '5411.T','5631.T','5706.T','5711.T','5713.T','5714.T','5801.T','5802.T',
    '5803.T','5831.T','6098.T','6146.T','6178.T','6273.T','6301.T','6302.T',
    '6305.T','6326.T','6361.T','6367.T','6471.T','6472.T','6473.T','6479.T',
    '6501.T','6503.T','6504.T','6506.T','6526.T','6594.T','6645.T','6674.T',
    '6701.T','6702.T','6723.T','6724.T','6752.T','6753.T','6758.T','6762.T',
    '6770.T','6841.T','6857.T','6861.T','6902.T','6920.T','6952.T','6954.T',
    '6971.T','6976.T','6981.T','6988.T','7004.T','7011.T','7012.T','7013.T',
    '7201.T','7202.T','7203.T','7205.T','7211.T','7261.T','7267.T','7269.T',
    '7270.T','7272.T','7453.T','7731.T','7733.T','7735.T','7741.T','7751.T',
    '7752.T','7762.T','7832.T','7911.T','7912.T','7951.T','7974.T','8001.T',
    '8002.T','8015.T','8031.T','8035.T','8053.T','8058.T','8233.T','8252.T',
    '8253.T','8267.T','8304.T','8306.T','8308.T','8309.T','8316.T','8331.T',
    '8354.T','8411.T','8591.T','8601.T','8604.T','8630.T','8697.T','8725.T',
    '8750.T','8766.T','8795.T','8801.T','8802.T','8804.T','8830.T','9001.T',
    '9005.T','9007.T','9008.T','9009.T','9020.T','9021.T','9022.T','9064.T',
    '9101.T','9104.T','9107.T','9147.T','9201.T','9202.T','9301.T','9432.T',
    '9433.T','9434.T','9501.T','9502.T','9503.T','9531.T','9532.T','9602.T',
    '9735.T','9766.T','9843.T','9983.T','9984.T'
]

BENCHMARK = "^N225"  # Nikkei225 index
OUT_DIR = os.getenv("BETA_OUT_DIR", os.path.join("reports", "beta"))
JQ_BASE = os.getenv("JQUANTS_BASE_URL", "https://api.jquants.com/v2").rstrip("/")
JQ_SLEEP_SEC = float(os.getenv("JQUANTS_SLEEP_SEC", "0.15"))


# -------------------------
# Helpers
# -------------------------
def normalize_code4(x: str) -> Optional[str]:
    s = str(x).replace(".T", "")
    s = re.sub(r"[^0-9]", "", s)
    # Some sources may use 5 digits with trailing 0 for preferred shares etc.
    if len(s) == 5 and s.endswith("0"):
        s = s[:-1]
    return s if len(s) == 4 else None


def jq_headers() -> Dict[str, str]:
    api_key = os.getenv("JQUANTS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("JQUANTS_API_KEY is not set in environment variables.")
    return {"x-api-key": api_key}


def jq_fetch_master_equities() -> pd.DataFrame:
    """
    Fetch /equities/master (v2) with pagination.
    """
    url = f"{JQ_BASE}/equities/master"
    headers = jq_headers()

    rows: List[Dict[str, Any]] = []
    next_token: Optional[str] = None
    params: Dict[str, Any] = {"limit": 2000}

    while True:
        if next_token:
            params["next_token"] = next_token
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        j = r.json()
        rows.extend(j.get("data", []) or [])

        next_token = j.get("next_token") or j.get("nextToken")
        if not next_token:
            break

        time.sleep(JQ_SLEEP_SEC)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_betas(prices_close: pd.DataFrame, tickers: List[str], bench: str) -> Tuple[pd.DataFrame, pd.Series, float]:
    """
    Compute:
      - log returns
      - per-ticker beta vs benchmark
    """
    rets = np.log(prices_close / prices_close.shift(1)).dropna(how="any")
    if bench not in rets.columns:
        raise RuntimeError(f"Benchmark {bench} not found in price data columns.")

    mkt = rets[bench].dropna()
    var_mkt = float(np.var(mkt))
    if not np.isfinite(var_mkt) or var_mkt <= 0:
        raise RuntimeError("Invalid market variance computed; cannot compute beta.")

    rows = []
    for t in tickers:
        if t not in rets.columns:
            continue
        j = pd.concat([rets[t], mkt], axis=1).dropna()
        if len(j) < 60:
            continue
        cov = np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1]
        beta = float(cov / var_mkt) if var_mkt else float("nan")
        if np.isfinite(beta):
            rows.append({"Ticker": t, "Beta": beta})

    beta_df = pd.DataFrame(rows)
    return beta_df, mkt, var_mkt


def compute_sector_betas(df_beta: pd.DataFrame, rets: pd.DataFrame, mkt: pd.Series, var_mkt: float) -> pd.DataFrame:
    sector_rows = []
    for s, g in df_beta.groupby("S17Nm"):
        if not s or (isinstance(s, float) and np.isnan(s)):
            continue
        mem = [t for t in g["Ticker"].tolist() if t in rets.columns]
        if len(mem) < 2:
            continue
        sr = rets[mem].mean(axis=1)
        j = pd.concat([sr, mkt], axis=1).dropna()
        if len(j) < 60:
            continue
        cov = np.cov(j.iloc[:, 0], j.iloc[:, 1])[0, 1]
        beta = float(cov / var_mkt) if var_mkt else float("nan")
        sector_rows.append({"S17Nm": s, "SectorBeta": beta, "Members": len(mem)})

    sector_beta = pd.DataFrame(sector_rows)
    return sector_beta


def save_outputs(df_beta: pd.DataFrame, sector_beta: pd.DataFrame) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")

    stock_latest = os.path.join(OUT_DIR, "beta_by_stock.csv")
    sector_latest = os.path.join(OUT_DIR, "beta_by_sector.csv")
    stock_hist = os.path.join(OUT_DIR, f"beta_by_stock_{stamp}.csv")
    sector_hist = os.path.join(OUT_DIR, f"beta_by_sector_{stamp}.csv")

    df_beta.to_csv(stock_latest, index=False, encoding="utf-8-sig")
    sector_beta.to_csv(sector_latest, index=False, encoding="utf-8-sig")
    df_beta.to_csv(stock_hist, index=False, encoding="utf-8-sig")
    sector_beta.to_csv(sector_hist, index=False, encoding="utf-8-sig")

    print("saved:")
    print(stock_latest)
    print(sector_latest)
    print(stock_hist)
    print(sector_hist)


def main() -> None:
    tickers = sorted(set(NIKKEI225_TICKERS))
    codes4 = sorted({normalize_code4(t) for t in tickers if normalize_code4(t)})

    # 1) J-Quants sector labels
    df_master = jq_fetch_master_equities()
    if df_master.empty:
        raise RuntimeError("J-Quants /equities/master returned empty data.")

    df_master["Code4"] = df_master["Code"].astype(str).str.replace(r"0$", "", regex=True)

    sector_map = (
        df_master[df_master["Code4"].isin(codes4)][["Code4", "CoName", "S17Nm", "S33Nm"]]
        .rename(columns={"Code4": "Code"})
        .copy()
    )

    # 2) Prices (1y, auto_adjust)
    prices = yf.download(
        tickers + [BENCHMARK],
        period="1y",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if prices is None or prices.empty:
        raise RuntimeError("yfinance download returned empty price data.")

    if isinstance(prices.columns, pd.MultiIndex):
        if "Close" not in prices.columns.get_level_values(0):
            raise RuntimeError("Close column not found in yfinance data (MultiIndex).")
        close = prices["Close"].copy()
    else:
        if "Close" not in prices.columns:
            raise RuntimeError("Close column not found in yfinance data.")
        close = prices[["Close"]].copy()

    if BENCHMARK not in close.columns:
        raise RuntimeError(f"Benchmark {BENCHMARK} not found in downloaded Close prices.")

    # 3) Individual betas
    beta_df, mkt, var_mkt = compute_betas(close, tickers, BENCHMARK)
    if beta_df.empty:
        raise RuntimeError("No betas computed (not enough data / missing tickers).")

    beta_df["Code"] = beta_df["Ticker"].str.replace(".T", "", regex=False)
    df_beta = beta_df.merge(sector_map, on="Code", how="left")

    # 4) Beta score (Z across all tickers)
    mean_b = float(df_beta["Beta"].mean())
    std_b = float(df_beta["Beta"].std())
    df_beta["BetaScore"] = (df_beta["Beta"] - mean_b) / std_b if std_b and np.isfinite(std_b) else float("nan")

    # 5) Sector betas (S17 average returns)
    rets = np.log(close / close.shift(1)).dropna(how="any")
    mkt = rets[BENCHMARK].dropna()
    sector_beta = compute_sector_betas(df_beta, rets, mkt, var_mkt)
    if not sector_beta.empty:
        sector_beta["SectorBetaScore"] = (sector_beta["SectorBeta"] - mean_b) / std_b if std_b and np.isfinite(std_b) else float("nan")

    # 6) Save
    df_beta = df_beta.sort_values("Beta", ascending=False).reset_index(drop=True)
    if not sector_beta.empty:
        sector_beta = sector_beta.sort_values("SectorBeta", ascending=False).reset_index(drop=True)

    save_outputs(df_beta, sector_beta)


if __name__ == "__main__":
    main()
