import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ====== 基本設定（既存運用に寄せる） ======
TZ_OFFSET = 9  # JST
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
FORCE_RUN = os.getenv("FORCE_RUN", "0") == "1"

# 取得範囲（この値は運用しながら調整OK）
DAILY_PERIOD = os.getenv("DAILY_PERIOD", "120d")      # 日足の取得期間
INTRADAY_PERIOD = os.getenv("INTRADAY_PERIOD", "14d") # 15分足の取得期間
INTRADAY_INTERVAL = os.getenv("INTRADAY_INTERVAL", "15m")

# レジーム判定（簡易）
VIX_LEVEL_ALERT = float(os.getenv("VIX_LEVEL_ALERT", "20"))
VIX_LEVEL_CRISIS = float(os.getenv("VIX_LEVEL_CRISIS", "25"))
VIX_SPIKE_15M_PCT = float(os.getenv("VIX_SPIKE_15M_PCT", "3.0"))
USDJPY_DROP_1D_PCT = float(os.getenv("USDJPY_DROP_1D_PCT", "-0.5"))       # 円高方向（USDJPY下落）
NIKKEI_FUT_DROP_1D_PCT = float(os.getenv("NIKKEI_FUT_DROP_1D_PCT", "-0.7"))


TICKERS = {
    "VIX": "^VIX",
    "USDJPY": "JPY=X",
    "NIKKEI": "^N225",
    "NIKKEI_FUT": "NK=F",
    "SPX": "^GSPC",
}

# ===== ユーティリティ（既存運用のまま寄せる） =====
def now_jst() -> datetime:
    return datetime.utcnow() + timedelta(hours=TZ_OFFSET)

def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5

def chunk_text(text: str, limit: int = 1900):
    out, buf, size = [], [], 0
    for line in text.splitlines():
        add = len(line) + 1
        if size + add > limit and buf:
            out.append("\n".join(buf))
            buf, size = [], 0
        buf.append(line)
        size += add
    if buf:
        out.append("\n".join(buf))
    return out

def discord_post(payload: dict):
    if not DISCORD_WEBHOOK_URL:
        print("[WARN] DISCORD_WEBHOOK_URL is empty. skip notify.", file=sys.stderr)
        return False
    try:
        r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=30)
        if r.status_code >= 300:
            print(f"[WARN] Discord post failed: {r.status_code} {r.text}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"[WARN] Discord post error: {e}", file=sys.stderr)
        return False

def discord_send_text(content: str):
    return discord_post({"content": content})

def fp(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"

# ===== データ取得（MultiIndex/tuple列も落ちないように） =====
def _flatten_columns(cols):
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(x) for x in c if x not in (None, "", " ")]
            out.append(parts[0] if parts else "")
        else:
            out.append(str(c))
    return out

def fetch_ohlc(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        prepost=False,
        threads=False,
        group_by="column",
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # indexは基本UTC想定 → JST相当で扱えるようにする（tz有無両対応）
    idx = df.index
    if getattr(idx, "tz", None) is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Asia/Tokyo")

    # 列名が tuple / MultiIndex の場合に備える
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    else:
        df.columns = _flatten_columns(df.columns)

    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df

def last_valid_close(df: pd.DataFrame) -> float:
    if df is None or df.empty or "close" not in df:
        return np.nan
    s = df["close"].dropna()
    return float(s.iloc[-1]) if len(s) else np.nan

def pct_change(a, b):
    if np.isnan(a) or np.isnan(b) or b == 0:
        return np.nan
    return (a / b - 1.0) * 100.0

def zscore_last(series: pd.Series, window: int = 20) -> float:
    s = series.dropna()
    if len(s) < max(5, window):
        return np.nan
    w = s.iloc[-window:]
    mu = w.mean()
    sd = w.std(ddof=0)
    if sd == 0:
        return np.nan
    return float((w.iloc[-1] - mu) / sd)

# ===== 特徴量作成（あなたのCSVと同列） =====
def build_features() -> pd.DataFrame:
    rows = []

    for name, sym in TICKERS.items():
        d = fetch_ohlc(sym, period=DAILY_PERIOD, interval="1d")
        i = fetch_ohlc(sym, period=INTRADAY_PERIOD, interval=INTRADAY_INTERVAL)

        d_close = last_valid_close(d)
        i_close = last_valid_close(i)

        # daily changes
        d_close_1d_ago = np.nan
        d_close_5d_ago = np.nan
        if d is not None and not d.empty and "close" in d:
            ds = d["close"].dropna()
            if len(ds) >= 2:
                d_close_1d_ago = float(ds.iloc[-2])
            if len(ds) >= 6:
                d_close_5d_ago = float(ds.iloc[-6])

        chg_1d = pct_change(d_close, d_close_1d_ago)
        chg_5d = pct_change(d_close, d_close_5d_ago)

        # intraday last 15m change
        chg_15m = np.nan
        if i is not None and not i.empty and "close" in i:
            iser = i["close"].dropna()
            if len(iser) >= 2:
                prev_i = float(iser.iloc[-2])
                chg_15m = pct_change(i_close, prev_i)

        # z-score on daily close (20d)
        z20 = np.nan
        if d is not None and not d.empty and "close" in d:
            z20 = zscore_last(d["close"], window=20)

        rows.append({
            "symbol": name,
            "daily_close": d_close,
            "daily_%chg_1d": chg_1d,
            "daily_%chg_5d": chg_5d,
            "intraday_close_15m": i_close,
            "intraday_%chg_last15m": chg_15m,
            "zscore_20d": z20,
        })

    return pd.DataFrame(rows)

# ===== レジーム判定（VIX主導＋円/先物で補強） =====
def eval_regime(feat: pd.DataFrame):
    def row(sym):
        x = feat[feat["symbol"] == sym]
        return x.iloc[0] if len(x) else None

    vix = row("VIX")
    usdjpy = row("USDJPY")
    nfut = row("NIKKEI_FUT")

    base = "NORMAL"
    if vix is not None and not np.isnan(vix["daily_close"]):
        if float(vix["daily_close"]) >= VIX_LEVEL_CRISIS:
            base = "CRISIS"
        elif float(vix["daily_close"]) >= VIX_LEVEL_ALERT:
            base = "ALERT"

    spike_flag = False
    if vix is not None and not np.isnan(vix["intraday_%chg_last15m"]):
        spike_flag = float(vix["intraday_%chg_last15m"]) >= VIX_SPIKE_15M_PCT

    riskoff_flags = []
    if usdjpy is not None and not np.isnan(usdjpy["daily_%chg_1d"]) and float(usdjpy["daily_%chg_1d"]) <= USDJPY_DROP_1D_PCT:
        riskoff_flags.append("USDJPY(円高)")
    if nfut is not None and not np.isnan(nfut["daily_%chg_1d"]) and float(nfut["daily_%chg_1d"]) <= NIKKEI_FUT_DROP_1D_PCT:
        riskoff_flags.append("NIKKEI_FUT下落")

    final_regime = base
    reasons = []
    if vix is not None:
        reasons.append(f"VIX={fp(vix['daily_close'],2)}")
    if spike_flag and vix is not None:
        reasons.append(f"VIX15m+{fp(vix['intraday_%chg_last15m'],2)}%")
    if riskoff_flags:
        reasons.append(" / ".join(riskoff_flags))

    if base == "NORMAL":
        if spike_flag or len(riskoff_flags) >= 2:
            final_regime = "ALERT"
    elif base == "ALERT":
        if spike_flag and len(riskoff_flags) >= 1:
            final_regime = "CRISIS"

    return final_regime, " | ".join(reasons) if reasons else "-"

# ===== 通知 =====
def notify(feat: pd.DataFrame):
    regime, reason = eval_regime(feat)
    ts = now_jst().strftime("%m/%d %H:%M")

    title = f"【Market Regime Monitor】{ts}  Regime={regime}"
    head = "symbol      daily_close  1d%     5d%     intra_close  15m%    z20"

    lines = [title, f"Reason: {reason}", head]

    # 表示順（あなたの表と同じ並び）
    order = ["VIX", "USDJPY", "NIKKEI", "NIKKEI_FUT", "SPX"]
    for sym in order:
        r = feat[feat["symbol"] == sym]
        if r.empty:
            lines.append(f"{sym:<10}  (no data)")
            continue
        r = r.iloc[0]
        lines.append(
            f"{sym:<10}  "
            f"{fp(r['daily_close'],2):>10}  "
            f"{fp(r['daily_%chg_1d'],3):>7}  "
            f"{fp(r['daily_%chg_5d'],3):>7}  "
            f"{fp(r['intraday_close_15m'],2):>10}  "
            f"{fp(r['intraday_%chg_last15m'],3):>7}  "
            f"{fp(r['zscore_20d'],3):>6}"
        )

    for part in chunk_text("\n".join(lines)):
        discord_send_text(part)

def main():
    now = now_jst()
    # 指数監視なので週末は基本スキップ（FORCE_RUN=1で強制）
    if not FORCE_RUN and is_weekend(now):
        print(f"[SKIP] {now:%F %R} weekend (FORCE_RUN=1 to override)")
        return

    feat = build_features()

    # 取得失敗でも落とさず通知
    if feat is None or feat.empty:
        discord_send_text(f"【Market Regime Monitor】{now_jst():%m/%d %H:%M}\nデータ取得失敗")
        return

    # CSVも出しておく（ActionsでArtifact化したい場合に便利）
    out_csv = os.getenv("OUT_CSV", "regime_features.csv")
    feat.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] saved: {out_csv}")

    notify(feat)

if __name__ == "__main__":
    main()
