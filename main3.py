# -*- coding: utf-8 -*-
# main2.py — 押し目抽出 → Discordへ「テキスト＋チャート画像」1通で送信
# 依存: pandas, numpy, yfinance, mplfinance, requests
# 環境変数:
#   PUBLIC_BASE_URL               （GitHub Pages のベースURL。例: https://<user>.github.io/<repo>）
#   DISCORD_WEBHOOK_URL          （DiscordのWebhook URL。未設定なら下の DEFAULT_WEBHOOK を使用）
#   （任意）FORCE_RUN=1         （週末スキップ無効化）
#   （任意）TICKERS_CSV=./tickers.csv  (Ticker列を含むCSV)
#   （任意）LOOKBACK_DAYS=180

import os
import sys
import math
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf

# ===== 設定（必要に応じて変更） =====
TZ_OFFSET = 9  # JST
REBOUND_MIN = 1.0       # 反発率 >= 1%   ←必要なら 2.0 に
REBOUND_MAX = 4.0       # 反発率 <= 4%
DROP_MAX = 15.0         # ピークからの許容下落率 <= 15%
DAYS_SINCE_MIN = 2      # 押し目から最新までの営業日数 >= 2
EXPECTED_RISE_MIN = 3.0 # 期待上昇率 >= 3%
SMA_WINDOW = 25
TOP_N = 15
DEFAULT_LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "180"))

# ===== 配信先 =====
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
DEFAULT_WEBHOOK = "https://canary.discord.com/api/webhooks/1410262330180243506/uyfezbFe3uTaMiqbQgY5d029FLtyyx4kkZf-iVHYJf6A9qeCl2w52b60mR57BBX5RXcN"
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", DEFAULT_WEBHOOK)

# ===== ユーティリティ =====
def now_jst():
    return datetime.utcnow() + timedelta(hours=TZ_OFFSET)

def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5  # 土日

# ===== RSI 計算ヘルパー =====
def latest_rsi_from_raw(raw_df, ticker: str, period: int = 14):
    """
    yf.download(..., group_by='column') の生データから対象ティッカーの終値でRSI(14)を算出。
    取得不可の場合は None を返す。
    """
    try:
        if isinstance(raw_df.columns, pd.MultiIndex):
            close = raw_df[("Close", ticker)].dropna()
        else:
            close = raw_df["Close"].dropna()
        if len(close) < period + 2:
            return None
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        roll_up = up.rolling(period, min_periods=period).mean()
        roll_down = down.rolling(period, min_periods=period).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi.iloc[-1])
    except Exception:
        return None

# ===== Discord 送信 =====
def discord_notify(text: str, image_url: str | None = None):
    """
    Discord Webhook に 1通で送る。
    - text は content として送信
    - image_url があれば embed の画像として添付
    """
    if not DISCORD_WEBHOOK_URL:
        print("[ERROR] DISCORD_WEBHOOK_URL is missing.", file=sys.stderr)
        return
    data = {"content": text}
    if image_url:
        data["embeds"] = [{"image": {"url": image_url}}]
    r = requests.post(DISCORD_WEBHOOK_URL, json=data, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"Discord送信失敗: {r.status_code} {r.text}")

# ===== データ取得 =====
def load_tickers():
    # 優先: 環境変数 TICKERS_CSV のCSV（Ticker列）
    csv_path = os.getenv("TICKERS_CSV")
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        col = None
        for c in df.columns:
            if str(c).lower() in ("ticker", "symbol", "code"):
                col = c
                break
        if col:
            tickers = [str(x).strip() for x in df[col].dropna().unique().tolist()]
            if tickers:
                return tickers
    # フォールバック：日経225の一部だけ（必要なら増やしてOK）
    return nikkei225_tickers

# ===== 日経225ティッカー（一部だけ記載、残り省略OK）=====
nikkei225_tickers = [
    "7203.T",  # トヨタ
    "6758.T",  # ソニーG
    "9984.T",  # ソフトバンクG
    "8035.T",  # 東エレク
    "6857.T",  # アドテスト
    # ... 必要に応じて追加 ...
]

# ===== 短縮名マップ（必要な分だけ）=====
ticker_name_map = {
    "7203.T": "トヨタ",
    "6758.T": "ソニーG",
    "9984.T": "ソフトバンクG",
    "8035.T": "東エレク",
    "6857.T": "アドテスト",
    # ... 必要に応じて追加 ...
}

def fetch_market_data(tickers, lookback_days=DEFAULT_LOOKBACK_DAYS):
    end_dt = (now_jst().date() + timedelta(days=1)).isoformat()
    start_dt = (now_jst().date() - timedelta(days=lookback_days)).isoformat()
    raw = yf.download(
        tickers,
        start=start_dt,
        end=end_dt,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",  # (field, ticker)
        threads=True,
    )
    # 必須カラムチェック
    for c in ("Close", "High", "Low"):
        if c not in raw.columns.get_level_values(0):
            raise RuntimeError(f"yfinance returned missing column: {c}")
    close = raw["Close"].copy()
    high = raw["High"].copy()
    low = raw["Low"].copy()
    return raw, close, high, low

# ===== 押し目抽出 =====
def rolling_sma(series: pd.Series, window=SMA_WINDOW):
    return series.rolling(window, min_periods=window).mean()

def compute_one_ticker(close_s: pd.Series, high_s: pd.Series, low_s: pd.Series, window_days=30):
    try:
        close_s = close_s.dropna()
        high_s = high_s.reindex_like(close_s).dropna()
        low_s  = low_s.reindex_like(close_s).dropna()
        if len(close_s) < max(SMA_WINDOW, window_days) + 2:
            return None

        # 対象期間
        look = close_s.iloc[-window_days:]
        look_high = high_s.loc[look.index]
        look_low  = low_s.loc[look.index]
        if look_high.empty or look_low.empty:
            return None

        # ピーク（期間内の最高値）
        peak_idx = look_high.idxmax()
        peak_val = float(look_high.loc[peak_idx])

        # ピーク後の最安値
        after_peak = look_low.loc[look_low.index > peak_idx]
        if after_peak.empty:
            return None
        pull_idx = after_peak.idxmin()
        pull_val = float(after_peak.loc[pull_idx])

        latest_idx = close_s.index[-1]
        latest_val = float(close_s.iloc[-1])
        prev_val = float(close_s.iloc[-2]) if len(close_s) >= 2 else np.nan
        sma25 = float(rolling_sma(close_s).iloc[-1]) if len(close_s) >= SMA_WINDOW else np.nan

        rebound_pct = (latest_val / pull_val - 1.0) * 100.0
        drop_pct = (1.0 - latest_val / peak_val) * 100.0
        expected_upper = float(peak_val)
        expected_rise_pct = (expected_upper / latest_val - 1.0) * 100.0
        days_since_pull = (close_s.index.get_loc(latest_idx) - close_s.index.get_loc(pull_idx))

        # 条件
        conds = [
            rebound_pct >= REBOUND_MIN,
            rebound_pct <= REBOUND_MAX,
            drop_pct <= DROP_MAX,
            days_since_pull >= DAYS_SINCE_MIN,
            not math.isnan(sma25) and latest_val >= sma25,
            expected_rise_pct >= EXPECTED_RISE_MIN,
            latest_val >= pull_val,
        ]
        if not all(conds):
            return None

        return {
            "Ticker": close_s.name,
            "Peak_Date": peak_idx.date(),
            "Peak_High": round(peak_val, 2),
            "Pullback_Date": pull_idx.date(),
            "Pullback_Low": round(pull_val, 2),
            "Latest_Date": latest_idx.date(),
            "Latest_Close": round(latest_val, 2),
            "Prev_Close": round(prev_val, 2) if not math.isnan(prev_val) else np.nan,
            "Return_%": round(expected_rise_pct, 2),
            "Rebound_From_Low_%": round(rebound_pct, 2),
            "Drop_From_Peak_%": round(drop_pct, 2),
            "Days_Since_Pullback": int(days_since_pull),
            "SMA25": round(sma25, 2) if not math.isnan(sma25) else np.nan,
            "Expected_Upper": round(expected_upper, 2),
            "Expected_Rise_%": round(expected_rise_pct, 2),
        }
    except Exception as e:
        print(f"[WARN] compute_one_ticker failed for {close_s.name}: {e}", file=sys.stderr)
        return None

def find_pullback_candidates(close_df: pd.DataFrame, high_df: pd.DataFrame, low_df: pd.DataFrame, window_days=30):
    rows = []
    for ticker in close_df.columns:
        res = compute_one_ticker(close_df[ticker], high_df[ticker], low_df[ticker], window_days=window_days)
        if res:
            rows.append(res)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("Return_%", ascending=False).reset_index(drop=True)

# ===== チャート画像作成 =====
def save_chart_image_from_raw(raw_df, ticker: str, out_dir="charts"):
    need_cols = ["Open", "High", "Low", "Close", "Volume"]
    try:
        use = raw_df.loc[:, [(c, ticker) for c in need_cols]].copy()
    except Exception:
        return None
    if use.empty:
        return None
    use.columns = need_cols
    use = use.dropna()
    if use.empty:
        return None

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.png")
    mpf.plot(
        use,
        type="candle",
        mav=(5, 25, 75),
        volume=True,
        style="yahoo",
        savefig=dict(fname=out_path, dpi=140, bbox_inches="tight")
    )
    return out_path

# ===== 名称辞書 =====
def build_ticker_name_map(tickers):
    return {t: ticker_name_map.get(t, "") for t in tickers}

# ===== パイプライン =====
def run_pipeline():
    tickers = load_tickers()
    raw, close, high, low = fetch_market_data(tickers, lookback_days=DEFAULT_LOOKBACK_DAYS)

    # 30日・14日で抽出 → マージ（同一ティッカーは 'Return_%' が大きい方を採用）
    rs = []
    for w in (30, 14):
        df = find_pullback_candidates(close, high, low, window_days=w)
        if not df.empty:
            df["Window"] = w
            rs.append(df)

    if not rs:
        return pd.DataFrame(), raw, {}

    cat = pd.concat(rs, ignore_index=True).sort_values(["Ticker", "Return_%"], ascending=[True, False])
    best = cat.groupby("Ticker", as_index=False).first().sort_values("Return_%", ascending=False).reset_index(drop=True)
    name_map = build_ticker_name_map(best["Ticker"].tolist())
    return best, raw, name_map

# ===== 通知（Discord: テキスト＋画像 1通）=====
def notify(best_df: pd.DataFrame, raw_df, ticker_name_map: dict, top_n=TOP_N):
    if best_df is None or best_df.empty:
        discord_notify("【押し目スクリーニング】本日は抽出なしでした。")
        return

    header = (
        f"📊【押し目スクリーニング】{now_jst().strftime('%m/%d %H:%M')}\n"
        f"抽出: {len(best_df)} 銘柄（重複統合）\n"
        f"条件: 反発≤{REBOUND_MAX:.0f}% & ≥{REBOUND_MIN:.0f}%・下落≤{DROP_MAX:.0f}%・SMA25上・期待≥{EXPECTED_RISE_MIN:.0f}%・{DAYS_SINCE_MIN}日経過\n"
        f"────────────────────"
    )
    discord_notify(header)

    for _, r in best_df.head(top_n).iterrows():
        ticker = str(r["Ticker"])
        name = ticker_name_map.get(ticker, "")
        upper  = r.get("Expected_Upper")
        latest = r.get("Latest_Close")
        low    = r.get("Pullback_Low")
        rise_p = r.get("Expected_Rise_%")
        prev   = r.get("Prev_Close")

        def fnum(x):
            try: return f"{float(x):,.0f}"
            except: return "-"
        def fpct(x):
            try: return f"{float(x):.1f}%"
            except: return "-"
        def fpct_signed(x):
            try:
                x = float(x)
                if not np.isfinite(x): return "-"
                return f"{x:+.1f}%"
            except:
                return "-"

        expect_amt = (float(upper) - float(latest)) if pd.notna(upper) and pd.notna(latest) else None
        chg_pct = ((float(latest) / float(prev)) - 1.0) * 100.0 if (pd.notna(latest) and pd.notna(prev) and float(prev) != 0.0) else None
        bot_pct = ((float(latest) / float(low)) - 1.0) * 100.0 if (pd.notna(latest) and pd.notna(low) and float(low) != 0.0) else None

        # 押し目記録日とRSI
        pull_date = r.get("Pullback_Date")
        pull_str = pull_date.strftime("%m/%d") if hasattr(pull_date, "strftime") else "-"
        rsi_val = latest_rsi_from_raw(raw_df, ticker, period=14)
        rsi_str = "-" if rsi_val is None or not np.isfinite(rsi_val) else f"{rsi_val:.0f}"

        # テキスト 4行（line1〜4）
        line1 = f"{ticker} {name}".rstrip()
        line2 = f"↗ {fpct(rise_p)}   🎯 上 {fnum(upper)}   下 {fnum(low)}"
        line3 = f"今 {fnum(latest)}   🎯 期待額 {fnum(expect_amt)}"
        line4 = f"変動率 {fpct_signed(chg_pct)}   底値比較 {fpct_signed(bot_pct)}   記録日 {pull_str}   RSI {rsi_str}"
        msg = "\n".join([line1, line2, line3, line4])

        # チャート画像を作成し、公開URLをDiscordに添付
        img_path = save_chart_image_from_raw(raw_df, ticker, out_dir="charts")
        if img_path and PUBLIC_BASE_URL:
            public_url = f"{PUBLIC_BASE_URL}/{os.path.basename(img_path)}"  # ← サイト直下に公開する想定
            discord_notify(msg, image_url=public_url)
        else:
            discord_notify(msg)

def main():
    now = now_jst()
    force = os.getenv("FORCE_RUN") == "1"

    if not force and is_weekend(now):
        print(f"[SKIP] {now:%F %R} 週末のためスキップ（FORCE_RUN=1で実行可能）")
        return

    best, raw, name_map = run_pipeline()
    notify(best, raw, name_map, top_n=TOP_N)

if __name__ == "__main__":
    main()
