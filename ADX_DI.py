# -*- coding: utf-8 -*-
"""
ADX + DI + ATR + BB(±1σ) screener for Nikkei225
+ Adds PER/PBR (yfinance info)
+ Adds J-Quants fundamentals (ROE proxy + simple earnings summary)
+ Adds AI comment (undervalued/fair/overvalued) using:
    - sector median PER/PBR (TSE 33-sector via J-Quants listed info)
    - stock PER/PBR
    - ROE (NP/Eq from J-Quants fins/summary)
    - latest earnings numbers (Sales/OP/NP and forecasts if available)

Design goals (per user policy):
- Objective numbers are computed in code.
- Final valuation label + brief reasoning is done by AI.
- AI is called ONLY when there are hits, and in ONE batch to reduce cost.
- J-Quants Free plan endpoints only (fins/summary + listed/info).
"""

import os
import sys
import json
import math
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ---------- chart libs ----------
import matplotlib
matplotlib.use("Agg")

# Optional: mplfinance (chart)
try:
    import mplfinance as mpf  # type: ignore
    MPF_AVAILABLE = True
except Exception:
    MPF_AVAILABLE = False

# Optional: OpenAI (for valuation comments)
try:
    from openai import OpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# =========================
# Config (Env)
# =========================
TZ_OFFSET = 9  # JST
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "180"))
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
FORCE_RUN = os.getenv("FORCE_RUN", "0") == "1"

# --- Screener params (kept close to your existing run) ---
ADX_MAX = 25.0
ATR_MIN_PCT = 1.8
ATR_MAX_PCT = 6.0
BB_TOUCH_MIN = 3
SMA_SLOPE_MAX_PCT = 5.0

DI_DIFF_MAX = float(os.getenv("DI_DIFF_MAX", "7.0"))
DI_RATIO_MIN = float(os.getenv("DI_RATIO_MIN", "1.03"))

# Output dirs
METRICS_OUT_DIR = os.getenv("METRICS_OUT_DIR", "reports")
METRICS_PREFIX = os.getenv("METRICS_PREFIX", "atr_swing_metrics")

CHART_OUT_DIR = os.getenv("CHART_OUT_DIR", "charts")
CHART_LOOKBACK_DAYS = int(os.getenv("CHART_LOOKBACK_DAYS", "90"))
CHART_TOP_N = int(os.getenv("CHART_TOP_N", "8"))

# --- J-Quants ---
JQUANTS_API_KEY = os.getenv("JQUANTS_API_KEY", "").strip()
JQUANTS_BASE = os.getenv("JQUANTS_BASE_URL", "https://api.jquants.com/v2").rstrip("/")

# --- AI ---
ENABLE_AI = os.getenv("ENABLE_AI", "1").strip() == "1"
AI_ONLY_ON_ALERT = os.getenv("AI_ONLY_ON_ALERT", "1").strip() == "1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "700"))
OPENAI_TIMEOUT_SEC = int(os.getenv("OPENAI_TIMEOUT_SEC", "30"))
OPENAI_RETRIES = int(os.getenv("OPENAI_RETRIES", "2"))
DEBUG_AI = os.getenv("DEBUG_AI", "0").strip() == "1"

# Rate limiting (safety)
JQ_SLEEP_SEC = float(os.getenv("JQUANTS_SLEEP_SEC", "0.15"))
JQ_TIMEOUT_SEC = int(os.getenv("JQUANTS_TIMEOUT_SEC", "20"))

# =========================
# Nikkei 225 tickers + Japanese names (existing mapping)
# =========================
# NOTE: Keep your current list; this file contains a placeholder minimal set.
# Replace/merge with your full nikkei225 list and name map as in your current ADX_DI.py.

# ===== 日経225ティッカー =====
nikkei225_tickers = [ '4151.T','4502.T','4503.T','4506.T','4507.T','4519.T','4523.T','4568.T','4578.T','6479.T','6501.T','6503.T','6504.T','6506.T','6526.T','6594.T','6645.T','6674.T','6701.T','6702.T','6723.T','6724.T','6752.T','6753.T','6758.T','6762.T','6770.T','6841.T','6857.T','6861.T','6902.T','6920.T','6952.T','6954.T','6971.T','6976.T','6981.T','7735.T','7751.T','7752.T','8035.T','7201.T','7202.T','7203.T','7205.T','7211.T','7261.T','7267.T','7269.T','7270.T','7272.T','4543.T','4902.T','6146.T','7731.T','7733.T','7741.T','7762.T','9432.T','9433.T','9434.T','6963.T','9984.T','5831.T','7186.T','8304.T','8306.T','8308.T','8309.T','8316.T','8331.T','8354.T','8411.T','8253.T','8591.T','8697.T','8601.T','8604.T','8630.T','8725.T','8750.T','8766.T','8795.T','1332.T','2002.T','2269.T','2282.T','2501.T','2502.T','2503.T','2801.T','2802.T','2871.T','2914.T','3086.T','3092.T','3099.T','3382.T','7453.T','8233.T','8252.T','8267.T','9843.T','9983.T','2413.T','2432.T','3659.T','4307.T','4324.T','4385.T','4661.T','4689.T','4704.T','4751.T','4755.T','6098.T','6178.T','7974.T','9602.T','9735.T','9766.T','1605.T','3401.T','3402.T','3861.T','3405.T','3407.T','4004.T','4005.T','4021.T','4042.T','4043.T','4061.T','4063.T','4183.T','4188.T','4208.T','4452.T','4901.T','4911.T','6988.T','5019.T','5020.T','5101.T','5108.T','5201.T','5214.T','5233.T','5301.T','5332.T','5333.T','5401.T','5406.T','5411.T','3436.T','5706.T','5711.T','5713.T','5714.T','5801.T','5802.T','5803.T','2768.T','8001.T','8002.T','8015.T','8031.T','8053.T','8058.T','1721.T','1801.T','1802.T','1803.T','1808.T','1812.T','1925.T','1928.T','1963.T','5631.T','6103.T','6113.T','6273.T','6301.T','6302.T','6305.T','6326.T','6361.T','6367.T','6471.T','6472.T','6473.T','7004.T','7011.T','7013.T','7012.T','7832.T','7911.T','7912.T','7951.T','3289.T','8801.T','8802.T','8804.T','8830.T','9001.T','9005.T','9007.T','9008.T','9009.T','9020.T','9021.T','9022.T','9064.T','9147.T','9101.T','9104.T','9107.T','9201.T','9202.T','9301.T','9501.T','9502.T','9503.T','9531.T','9532.T' ]

# ===== 短縮名マップ =====
ticker_name_map = {
    "1332.T": "日水", "1333.T": "マルハニチロ", "1605.T": "INPEX", "1801.T": "大成建",
    "1802.T": "清水建", "1803.T": "飛島建", "1808.T": "長谷工", "1812.T": "鹿島",
    "1925.T": "大和ハウス", "1928.T": "積水ハウス", "1963.T": "日揮HD", "2002.T": "日清粉G",
    "2269.T": "明治HD", "2282.T": "日本ハム", "2413.T": "エムスリー", "2432.T": "DeNA",
    "2501.T": "サッポロHD", "2502.T": "アサヒGHD", "2503.T": "キリンHD", "2768.T": "双日",
    "2801.T": "キッコマン", "2802.T": "味の素", "2871.T": "ニチレイ", "2914.T": "JT",
    "3086.T": "Jフロント", "3092.T": "ZOZO", "3099.T": "三越伊勢丹", "3382.T": "セブン&アイ",
    "3401.T": "帝人", "3402.T": "東レ", "3405.T": "クラレ", "3407.T": "旭化成",
    "3436.T": "SUMCO", "3861.T": "王子HD", "4004.T": "昭電工", "4005.T": "住友化学",
    "4021.T": "日産化", "4042.T": "東ソー", "4043.T": "トクヤマ", "4061.T": "電化",
    "4063.T": "信越化", "4183.T": "三井化学", "4188.T": "三菱ケミHD", "4208.T": "UBE",
    "4452.T": "花王", "4502.T": "武田薬品", "4503.T": "アステラス", "4506.T": "大日本住友",
    "4507.T": "塩野義", "4519.T": "中外製薬", "4523.T": "エーザイ", "4543.T": "テルモ",
    "4568.T": "第一三共", "4578.T": "大塚HD", "4661.T": "OLC", "4689.T": "ZHD",
    "4704.T": "トレンド", "4751.T": "サイバー", "4755.T": "楽天G", "4901.T": "富士フイルム",
    "4902.T": "コニカミノルタ", "4911.T": "資生堂", "5020.T": "ENEOS",
    "5101.T": "横浜ゴム", "5108.T": "ブリヂストン", "5201.T": "AGC", "5214.T": "日電硝",
    "5233.T": "太平洋セメ", "5301.T": "東海カーボン", "5332.T": "TOTO", "5333.T": "日本ガイシ",
    "5401.T": "日本製鉄", "5406.T": "神戸製鋼", "5411.T": "JFEHD", "5706.T": "三井金属",
    "5711.T": "三菱マテ", "5713.T": "住友金属鉱山", "5714.T": "DOWA", "5801.T": "古河電工",
    "5802.T": "住友電工", "5803.T": "フジクラ", "6098.T": "リクルートHD", "6178.T": "日本郵政",
    "6273.T": "SMC", "6301.T": "コマツ", "6302.T": "住友重機", "6305.T": "日立建機",
    "6326.T": "クボタ", "6361.T": "荏原", "6367.T": "ダイキン", "6471.T": "日精工",
    "6472.T": "NTN", "6473.T": "ジェイテクト", "6479.T": "ミネベアミツミ", "6501.T": "日立",
    "6503.T": "三菱電機", "6504.T": "富士電機", "6506.T": "安川電機", "6526.T": "ソシオネクスト",
    "6594.T": "日電産", "6645.T": "オムロン", "6674.T": "ジーエスユアサ", "6701.T": "NEC",
    "6702.T": "富士通", "6723.T": "ルネサス", "6724.T": "セイコーエプソン", "6752.T": "パナソニック",
    "6753.T": "シャープ", "6758.T": "ソニーG", "6762.T": "TDK", "6770.T": "アルプスアルパ",
    "6841.T": "横河電機", "6857.T": "アドテスト", "6861.T": "キーエンス", "6902.T": "デンソー",
    "6920.T": "レーザーテック", "6952.T": "カシオ", "6954.T": "ファナック", "6971.T": "京セラ",
    "6976.T": "太陽誘電", "6981.T": "村田製作所", "6988.T": "日東電工", "7201.T": "日産自",
    "7202.T": "いすゞ", "7203.T": "トヨタ", "7205.T": "日野自", "7211.T": "三菱自",
    "7261.T": "マツダ", "7267.T": "ホンダ", "7269.T": "スズキ", "7270.T": "SUBARU",
    "7272.T": "ヤマハ発", "7453.T": "良品計画", "7731.T": "ニコン", "7733.T": "オリンパス",
    "7735.T": "スクリン", "7741.T": "HOYA", "7751.T": "キヤノン", "7752.T": "リコー",
    "7762.T": "シチズン", "7832.T": "バンナムHD", "7911.T": "凸版印刷", "7912.T": "大日本印刷",
    "7951.T": "ヤマハ", "7974.T": "任天堂", "8001.T": "伊藤忠", "8002.T": "丸紅",
    "8015.T": "豊田通商", "8031.T": "三井物産", "8035.T": "東エレク", "8053.T": "住友商事",
    "8058.T": "三菱商事", "8113.T": "ユニチャーム", "8252.T": "丸井G", "8253.T": "クレセゾン",
    "8267.T": "イオン", "8304.T": "あおぞら銀", "8306.T": "三菱UFJ", "8308.T": "りそなHD",
    "8309.T": "三井住友", "8316.T": "三井住友信託", "8331.T": "千葉銀", "8354.T": "ふくおかFG",
    "8411.T": "みずほ", "8591.T": "オリックス", "8601.T": "大和証G", "8604.T": "野村HD",
    "8630.T": "住友信託", "8697.T": "日取所", "8725.T": "MS&AD", "8750.T": "第一生命",
    "8766.T": "東京海上", "8795.T": "T&DHD", "8801.T": "三井不", "8802.T": "三菱地所",
    "8804.T": "東京建物", "8830.T": "住友不", "9001.T": "東武", "9005.T": "東急",
    "9007.T": "小田急", "9008.T": "京王", "9009.T": "京成", "9020.T": "JR東日本",
    "9021.T": "JR西日本", "9022.T": "JR東海", "9064.T": "ヤマトHD", "9101.T": "日本郵船",
    "9104.T": "商船三井", "9107.T": "川崎汽船", "9147.T": "NXHD", "9201.T": "JAL",
    "9202.T": "ANAHD", "9301.T": "三菱倉庫", "9432.T": "NTT", "9433.T": "KDDI",
    "9434.T": "ソフトバンク", "9501.T": "東電HD", "9502.T": "中部電", "9503.T": "関西電",
    "9531.T": "東ガス", "9532.T": "大阪ガス", "9602.T": "東宝", "9613.T": "NTTデータ",
    "9735.T": "セコム", "9766.T": "コナミG", "9843.T": "ニトリHD", "9983.T": "ファーストリテ",
    "9984.T": "ソフトバンクG",
}

# =========================
# Utils
# =========================
def now_jst() -> datetime:
    return datetime.utcnow() + timedelta(hours=TZ_OFFSET)

def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5

def chunk_text(text: str, limit: int = 1900) -> List[str]:
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

def fp(x, nd=2) -> str:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "-"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def diff_pct(val: Optional[float], base: Optional[float]) -> Optional[float]:
    """Return (val/base - 1) * 100, or None."""
    try:
        if val is None or base is None:
            return None
        if isinstance(val, float) and math.isnan(val):
            return None
        if isinstance(base, float) and math.isnan(base):
            return None
        if base == 0:
            return None
        return (float(val) / float(base) - 1.0) * 100.0
    except Exception:
        return None

def fmt_vs_median(label: str, val: Optional[float], med: Optional[float], nd_val=2, nd_med=2) -> str:
    d = diff_pct(val, med)
    d_str = "-" if d is None else f"{d:+.0f}%"
    return f"{label}:{fp(val, nd_val)} vs 業種中央値 {fp(med, nd_med)}（{d_str}）"

def fair_price(val_mul: Optional[float], target: Optional[float]) -> Optional[float]:
    try:
        if val_mul is None or target is None:
            return None
        if isinstance(val_mul, float) and math.isnan(val_mul):
            return None
        if isinstance(target, float) and math.isnan(target):
            return None
        return float(val_mul) * float(target)
    except Exception:
        return None

def fmt_fair(title: str, price: Optional[float], fair: Optional[float]) -> str:
    d = diff_pct(price, fair)
    d_str = "-" if d is None else f"{d:+.0f}%"
    return f"{title}:{fp(fair,0)}円（{d_str}）"

def discord_post(payload: dict, files=None) -> bool:
    if not DISCORD_WEBHOOK_URL:
        print("[WARN] DISCORD_WEBHOOK_URL is empty. skip notify.", file=sys.stderr)
        return False
    try:
        if files:
            r = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files, timeout=30)
        else:
            r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=30)
        if r.status_code >= 300:
            print(f"[WARN] Discord post failed: {r.status_code} {r.text}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"[WARN] Discord post error: {e}", file=sys.stderr)
        return False

def discord_send_text(content: str) -> bool:
    return discord_post({"content": content})

def discord_send_image_file(file_path: str, title: str = "", description: str = "") -> bool:
    if not os.path.exists(file_path):
        return False
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        payload = {"content": f"**{title}**\n{description}".strip()}
        return discord_post(payload, files=files)

# =========================
# Data fetch: yfinance
# =========================
def fetch_market_data(tickers: List[str], lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    end = now_jst().date()
    start = end - timedelta(days=lookback_days + 10)
    raw = yf.download(
        tickers=tickers,
        start=str(start),
        end=str(end + timedelta(days=1)),
        interval="1d",
        group_by="column",
        auto_adjust=False,
        threads=True,
        progress=False,
        actions=True,
    )
    return raw

# =========================
# Technical indicators
# =========================
def _series_from_raw(raw_df: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    if isinstance(raw_df.columns, pd.MultiIndex):
        return raw_df[(field, ticker)].dropna()
    return raw_df[field].dropna()

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def calc_adx_di(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    atr = tr.rolling(period, min_periods=period).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).rolling(period, min_periods=period).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).rolling(period, min_periods=period).mean() / atr

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = dx.rolling(period, min_periods=period).mean()
    return adx, plus_di, minus_di

def calc_bb(close: pd.Series, period: int = 20, sigma: float = 1.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std()
    upper = ma + sigma * sd
    lower = ma - sigma * sd
    return ma, upper, lower

def sma_slope_pct(sma: pd.Series, window: int = 20) -> float:
    if len(sma) < window + 1 or pd.isna(sma.iloc[-1]) or pd.isna(sma.iloc[-window-1]):
        return float("nan")
    base = float(sma.iloc[-window-1])
    if base == 0:
        return float("nan")
    return (float(sma.iloc[-1]) / base - 1.0) * 100.0

def bb_touch_count(close: pd.Series, upper: pd.Series, lower: pd.Series, window: int = 20) -> int:
    c = close.tail(window)
    u = upper.tail(window)
    l = lower.tail(window)
    hit = ((c >= u) | (c <= l)).sum()
    return int(hit)

def calc_latest_metrics(raw_df: pd.DataFrame, ticker: str) -> Optional[Dict[str, Any]]:
    try:
        close = _series_from_raw(raw_df, "Close", ticker)
        high = _series_from_raw(raw_df, "High", ticker)
        low = _series_from_raw(raw_df, "Low", ticker)

        need_len = 60
        if len(close) < need_len:
            return None

        adx, plus_di, minus_di = calc_adx_di(high, low, close, 14)
        atr20 = calc_atr(high, low, close, 20)

        sma25 = close.rolling(25, min_periods=25).mean()
        sma_slope = sma_slope_pct(sma25, window=20)

        bb_ma, bb_up, bb_dn = calc_bb(close, 20, sigma=1.0)
        touches = bb_touch_count(close, bb_up, bb_dn, window=20)

        last_close = float(close.iloc[-1])
        atr_pct = float(atr20.iloc[-1] / last_close * 100.0) if not pd.isna(atr20.iloc[-1]) else float("nan")

        adx_v = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else float("nan")
        pdi = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else float("nan")
        mdi = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else float("nan")

        di_diff = abs(pdi - mdi) if (not math.isnan(pdi) and not math.isnan(mdi)) else float("nan")
        di_ratio = (pdi / mdi) if (not math.isnan(pdi) and not math.isnan(mdi) and mdi != 0) else float("nan")

        bb_up_1 = float(bb_up.iloc[-1]) if not pd.isna(bb_up.iloc[-1]) else float("nan")
        bb_dn_1 = float(bb_dn.iloc[-1]) if not pd.isna(bb_dn.iloc[-1]) else float("nan")
        bottom_rise_ratio = (bb_up_1 / bb_dn_1) if (not math.isnan(bb_up_1) and not math.isnan(bb_dn_1) and bb_dn_1 != 0) else float("nan")

        passed = (
            (not math.isnan(adx_v) and adx_v <= ADX_MAX) and
            (not math.isnan(atr_pct) and (ATR_MIN_PCT <= atr_pct <= ATR_MAX_PCT)) and
            (touches >= BB_TOUCH_MIN) and
            (not math.isnan(sma_slope) and abs(sma_slope) <= SMA_SLOPE_MAX_PCT) and
            (not math.isnan(di_diff) and di_diff <= DI_DIFF_MAX) and
            (not math.isnan(di_ratio) and di_ratio >= DI_RATIO_MIN)
        )

        return {
            "Ticker": ticker,
            "Name": ticker_name_map.get(ticker, ""),
            "Close": last_close,
            "ATR_pct": atr_pct,
            "ADX": adx_v,
            "DI_diff": di_diff,
            "DI_ratio": di_ratio,
            "BB_touches": touches,
            "BB_dn1": bb_dn_1,
            "BB_up1": bb_up_1,
            "Bottom_Rise_Ratio": bottom_rise_ratio,
            "SMA_slope_pct": sma_slope,
            "Pass": bool(passed),
        }
    except Exception:
        return None

def compute_all_metrics(raw_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        m = calc_latest_metrics(raw_df, t)
        if m:
            rows.append(m)
    return pd.DataFrame(rows)

def screen_candidates(raw_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    df = compute_all_metrics(raw_df, tickers)
    if df.empty:
        return df
    return df[df["Pass"] == True].sort_values(["ADX", "ATR_pct"], ascending=[True, True]).reset_index(drop=True)

# =========================
# PER / PBR from yfinance
# =========================
def fetch_per_pbr_from_info(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        info = yf.Ticker(ticker).info or {}
        per = info.get("trailingPE", None)
        pbr = info.get("priceToBook", None)
        per = float(per) if per is not None else None
        pbr = float(pbr) if pbr is not None else None
        return per, pbr
    except Exception:
        return None, None

# =========================
# J-Quants helpers (Free endpoints)
# =========================
def _jq_headers() -> Dict[str, str]:
    if not JQUANTS_API_KEY:
        return {}
    return {"x-api-key": JQUANTS_API_KEY}

def jq_get(path: str, params: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    if not JQUANTS_API_KEY:
        return None
    url = f"{JQUANTS_BASE}{path}"
    try:
        r = requests.get(url, headers=_jq_headers(), params=params or {}, timeout=JQ_TIMEOUT_SEC)
        if r.status_code >= 300:
            if DEBUG_AI:
                print(f"[DEBUG] JQ GET failed: {r.status_code} {r.text[:200]}", file=sys.stderr)
            return None
        return r.json()
    except Exception as e:
        if DEBUG_AI:
            print(f"[DEBUG] JQ GET exception: {e}", file=sys.stderr)
        return None

def to_jq_code(ticker: str) -> str:
    """'6841.T' -> '6841' (API accepts 4-digit; response may include 5-digit with trailing 0)"""
    return ticker.split(".")[0]

def normalize_code(code: str) -> str:
    """'68410' -> '6841' (common J-Quants equity code formatting)"""
    s = str(code or "").strip()
    if len(s) == 5 and s.endswith("0"):
        return s[:-1]
    return s

def jq_fetch_fins_summary(code4: str) -> Optional[Dict[str, Any]]:
    """Return most recent record for given code (4-digit string)."""
    js = jq_get("/fins/summary", params={"code": code4})
    if not js or "data" not in js or not js["data"]:
        return None
    # data is list; choose newest by DiscDate/DiscTime
    data = js["data"]
    def key(x):
        return (x.get("DiscDate") or "", x.get("DiscTime") or "")
    data_sorted = sorted(data, key=key, reverse=True)
    return data_sorted[0]

def jq_fetch_listed_info_all() -> Optional[pd.DataFrame]:
    """
    Get listed issue master.
    V2 docs: /listed/info (GitBook) - returns many rows.
    We'll fetch once per run and filter to Nikkei225 codes.
    """
    js = jq_get("/listed/info", params={})
    if not js:
        return None
    # Some responses use 'info', others 'data'. Be robust.
    rows = js.get("info") or js.get("data") or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df

def build_sector_map_and_medians(tickers: List[str], per_pbr: Dict[str, Tuple[Optional[float], Optional[float]]]) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    """
    Returns:
      sector_map: ticker -> sector33code (string)
      medians: sector33code -> {'per_median': x, 'pbr_median': y}
    """
    sector_map: Dict[str, str] = {}
    medians: Dict[str, Dict[str, float]] = {}

    df = jq_fetch_listed_info_all()
    if df is None or df.empty:
        return sector_map, medians

    # Normalize code columns
    # Expect 'Code' and 'Sector33Code' (per docs)
    if "Code" not in df.columns or "Sector33Code" not in df.columns:
        return sector_map, medians

    df["Code4"] = df["Code"].astype(str).map(normalize_code)
    needed_codes = set([to_jq_code(t) for t in tickers])
    df_use = df[df["Code4"].isin(needed_codes)].copy()
    if df_use.empty:
        return sector_map, medians

    # Map
    for _, r in df_use.iterrows():
        code4 = str(r.get("Code4", "")).strip()
        s33 = str(r.get("Sector33Code", "")).strip()
        if not code4 or not s33:
            continue
        # find tickers matching this code
        for t in tickers:
            if to_jq_code(t) == code4:
                sector_map[t] = s33

    # Build per-sector median
    rows = []
    for t in tickers:
        s33 = sector_map.get(t, "")
        per, pbr = per_pbr.get(t, (None, None))
        if not s33:
            continue
        if per is None and pbr is None:
            continue
        rows.append({"Sector33Code": s33, "Ticker": t, "PER": per, "PBR": pbr})
    if not rows:
        return sector_map, medians

    d = pd.DataFrame(rows)
    for s33, g in d.groupby("Sector33Code"):
        per_med = g["PER"].dropna().median() if g["PER"].notna().any() else float("nan")
        pbr_med = g["PBR"].dropna().median() if g["PBR"].notna().any() else float("nan")
        medians[str(s33)] = {
            "per_median": float(per_med) if not pd.isna(per_med) else float("nan"),
            "pbr_median": float(pbr_med) if not pd.isna(pbr_med) else float("nan"),
        }
    return sector_map, medians

# =========================
# AI valuation comment (batch)
# =========================
def openai_client() -> Optional["OpenAI"]:
    if not (OPENAI_AVAILABLE and OPENAI_API_KEY):
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

def call_openai_text(client: "OpenAI", system: str, user: str) -> Optional[str]:
    for attempt in range(1 + max(0, OPENAI_RETRIES)):
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=OPENAI_MAX_TOKENS,
                timeout=OPENAI_TIMEOUT_SEC,
            )
            text = (resp.choices[0].message.content or "").strip()
            if DEBUG_AI:
                print("[DEBUG] AI raw head:", text[:1000])
            if text:
                return text
        except Exception as e:
            print(f"[WARN] OpenAI failed (attempt {attempt+1}): {e}", file=sys.stderr)
            time.sleep(0.6)
    return None

def ai_build_valuation_comments(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    """
    Input rows: list of candidate dicts with per/pbr + sector medians + roe + earnings summary.
    Output: {ticker: {"label": "...", "reason": "..."}}
    """
    if not ENABLE_AI:
        return {}
    if AI_ONLY_ON_ALERT and not rows:
        return {}
    client = openai_client()
    if client is None:
        return {}

    system = (
        "You are an equity valuation assistant for Japanese stocks.\n"
        "Given sector median PER/PBR (TSE 33-sector), stock PER/PBR, ROE, and latest earnings summary, "
        "output for each stock a label in Japanese: '割安', '妥当', or '割高', "
        "and a very brief reason (1-2 lines) in Japanese.\n"
        "Do NOT mention thresholds. Base your judgement on relative comparison + profitability + earnings tone.\n"
        "Return STRICT JSON only in the following format:\n"
        "{ \"TICKER\": {\"label\":\"割安|妥当|割高\", \"reason\":\"...\"}, ... }\n"
    )
    payload = {
        "date_jst": now_jst().strftime("%Y-%m-%d %H:%M"),
        "stocks": rows,
        "notes": "sector medians are computed as medians over Nikkei225 members within each Sector33Code.",
    }
    user = json.dumps(payload, ensure_ascii=False)
    text = call_openai_text(client, system, user)
    if not text:
        return {}
    # Try to parse JSON safely
    try:
        # Remove code fences if any
        text2 = text.strip()
        if text2.startswith("```"):
            text2 = re.sub(r"^```[a-zA-Z0-9]*\n", "", text2)
            text2 = re.sub(r"\n```$", "", text2)
        obj = json.loads(text2)
        out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    out[str(k)] = {
                        "label": str(v.get("label", "")).strip(),
                        "reason": str(v.get("reason", "")).strip(),
                    }
        return out
    except Exception:
        return {}

# =========================
# Earnings summary builder (from J-Quants fins/summary)
# =========================
def build_earnings_summary(jq_row: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not jq_row:
        return {}
    def s2f(x) -> Optional[float]:
        try:
            if x is None or x == "":
                return None
            return float(x)
        except Exception:
            return None

    sales = s2f(jq_row.get("Sales"))
    op = s2f(jq_row.get("OP"))
    np_ = s2f(jq_row.get("NP"))
    eps = s2f(jq_row.get("EPS"))
    eq = s2f(jq_row.get("Eq"))
    sh_out = s2f(jq_row.get("ShOutFY"))  # shares outstanding (FY)
    bps = (eq / sh_out) if (eq is not None and sh_out not in (None, 0.0)) else None
    roe = (np_ / eq) if (np_ is not None and eq not in (None, 0.0)) else None

    fsales = s2f(jq_row.get("FSales"))
    fop = s2f(jq_row.get("FOP"))
    fnp = s2f(jq_row.get("FNP"))

    return {
        "DiscDate": jq_row.get("DiscDate", ""),
        "CurPerType": jq_row.get("CurPerType", ""),
        "CurPerSt": jq_row.get("CurPerSt", ""),
        "CurPerEn": jq_row.get("CurPerEn", ""),
        "Sales": sales,
        "OP": op,
        "NP": np_,
        "EPS": eps,
        "Eq": eq,
        "ShOutFY": sh_out,
        "BPS_approx": bps,
        "ROE_proxy": roe,
        "Forecast": {"Sales": fsales, "OP": fop, "NP": fnp},
    }

# =========================
# Chart (simple)
# =========================
def save_chart_image_with_bb1sigma(raw_df: pd.DataFrame, ticker: str, out_dir: str) -> Optional[str]:
    if not MPF_AVAILABLE:
        return None
    try:
        os.makedirs(out_dir, exist_ok=True)
        close = _series_from_raw(raw_df, "Close", ticker)
        high = _series_from_raw(raw_df, "High", ticker)
        low = _series_from_raw(raw_df, "Low", ticker)
        open_ = _series_from_raw(raw_df, "Open", ticker)
        vol = _series_from_raw(raw_df, "Volume", ticker)

        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}).dropna()
        df = df.tail(CHART_LOOKBACK_DAYS)

        # BB ±1σ
        ma, up, dn = calc_bb(df["Close"], 20, sigma=1.0)
        apds = [
            mpf.make_addplot(ma, panel=0),
            mpf.make_addplot(up, panel=0),
            mpf.make_addplot(dn, panel=0),
        ]

        path = os.path.join(out_dir, f"{ticker.replace('.','_')}_bb1s.png")
        mpf.plot(
            df,
            type="candle",
            volume=True,
            addplot=apds,
            style="yahoo",
            figsize=(10, 6),
            title=f"{ticker}",
            savefig=dict(fname=path, dpi=140, bbox_inches="tight"),
        )
        return path
    except Exception as e:
        print(f"[WARN] chart failed {ticker}: {e}", file=sys.stderr)
        return None

# =========================
# Notify
# =========================
def notify(df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    ts = now_jst().strftime("%m/%d %H:%M")
    title = "【ATRレンジ候補（ADX≤25 × ATR% × BB±1σ × SMA × DI）】"
    if df is None or df.empty:
        discord_send_text(f"{title} {ts}\n該当なし")
        return

    # --- Enrich PER/PBR for all tickers in Nikkei225 (for sector medians) ---
    per_pbr_all: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for t in nikkei225_tickers:
        per_pbr_all[t] = fetch_per_pbr_from_info(t)

    sector_map, sector_medians = build_sector_map_and_medians(nikkei225_tickers, per_pbr_all)

    # --- Candidate enrich: PER/PBR + JQ + sector medians ---
    candidates: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        t = r["Ticker"]
        per, pbr = per_pbr_all.get(t, (None, None))
        code4 = to_jq_code(t)
        jq_row = jq_fetch_fins_summary(code4)
        earn = build_earnings_summary(jq_row)

        s33 = sector_map.get(t, "")
        med = sector_medians.get(s33, {})
        per_med = med.get("per_median", float("nan"))
        pbr_med = med.get("pbr_median", float("nan"))

        cand = {
            "ticker": t,
            "name": ticker_name_map.get(t, ""),
            "price": float(r.get("Close", float("nan"))),
            "per": per,
            "pbr": pbr,
            "sector33": s33,
            "sector_per_median": None if math.isnan(per_med) else per_med,
            "sector_pbr_median": None if math.isnan(pbr_med) else pbr_med,
            "roe": earn.get("ROE_proxy"),
            "earnings": {
                "disc_date": earn.get("DiscDate"),
                "period": earn.get("CurPerType"),
                "sales": earn.get("Sales"),
                "op": earn.get("OP"),
                "np": earn.get("NP"),
                "eps": earn.get("EPS"),
                "bps": earn.get("BPS_approx"),
                "forecast": earn.get("Forecast", {}),
            },
            "tech": {
                "atr_pct": float(r.get("ATR_pct", float("nan"))),
                "adx": float(r.get("ADX", float("nan"))),
                "di_diff": float(r.get("DI_diff", float("nan"))),
                "bb_touches": int(r.get("BB_touches", 0)),
                "sma_slope_pct": float(r.get("SMA_slope_pct", float("nan"))),
            },
        }
        candidates.append(cand)
        time.sleep(JQ_SLEEP_SEC)

    # --- AI comment batch (only for hits) ---
    ai_map: Dict[str, Dict[str, str]] = {}
    if candidates and ENABLE_AI and (not AI_ONLY_ON_ALERT or True):
        ai_map = ai_build_valuation_comments(candidates)

    # --- Text summary message (compact; no duplicated table section) ---
    lines = [f"{title} {ts}", f"件数: {len(df)}"]
    for i, r in enumerate(df.head(20).itertuples(index=False), start=1):
        t = getattr(r, "Ticker")
        name = ticker_name_map.get(t, "")
        per, pbr = per_pbr_all.get(t, (None, None))
        ai = ai_map.get(t, {})
        label = ai.get("label", "")
        lines.append(
            f"{i}. {t} {name}  価格:{fp(getattr(r,'Close',None),0)}  PER:{fp(per,2)} PBR:{fp(pbr,2)}"
            f"  {label}"
        )
    for msg in chunk_text("\n".join(lines)):
        discord_send_text(msg)

    # --- Send charts with rich description (includes AI reason + fundamentals) ---
    top = df.head(CHART_TOP_N)
    for _, rr in top.iterrows():
        t = rr["Ticker"]
        name = ticker_name_map.get(t, "")
        per, pbr = per_pbr_all.get(t, (None, None))

        # build desc
        ai = ai_map.get(t, {})
        label = ai.get("label", "")
        reason = ai.get("reason", "")

        # find candidate enriched
        c = next((x for x in candidates if x["ticker"] == t), None) or {}
        sector_per = c.get("sector_per_median")
        sector_pbr = c.get("sector_pbr_median")
        roe = c.get("roe")
        ed = (c.get("earnings") or {})
        disc = ed.get("disc_date") or "-"
        period = ed.get("period") or "-"
        sales = ed.get("sales")
        op = ed.get("op")
        np_ = ed.get("np")
        price = float(rr.get("Close", float("nan")))

        desc_lines = []
        if label:
            desc_lines.append(f"【評価】{label}  {reason}".strip())
        # --- fair value (sector median based) ---
        eps = (ed.get("eps") if isinstance(ed, dict) else None)
        bps = (ed.get("bps") if isinstance(ed, dict) else None)
        fair_pbr = fair_price(bps, sector_pbr)
        fair_per = fair_price(eps, sector_per)

        desc_lines.append(fmt_fair("理論株価(PBR基準)", price, fair_pbr))
        desc_lines.append(fmt_fair("理論株価(PER基準)", price, fair_per))
        desc_lines.append(
            f"{fmt_vs_median('PBR', pbr, sector_pbr, 2, 2)} / {fmt_vs_median('PER', per, sector_per, 2, 2)} / ROE(簡易):{fp(roe*100 if roe is not None else None,1)}%"
        )
        desc_lines.append(
            f"決算:{disc} {period}  売上:{fp(sales/1e8 if sales else None,1)}億  営業益:{fp(op/1e8 if op else None,1)}億  純益:{fp(np_/1e8 if np_ else None,1)}億"
        )
        desc_lines.append(
            f"ATR%:{fp(rr.get('ATR_pct'),2)} ADX:{fp(rr.get('ADX'),1)} DIΔ:{fp(rr.get('DI_diff'),1)} SMA傾き%:{fp(rr.get('SMA_slope_pct'),2)}"
        )
        desc = "\n".join([s for s in desc_lines if s.strip()])

        img = save_chart_image_with_bb1sigma(raw_df, t, out_dir=CHART_OUT_DIR)
        if img:
            discord_send_image_file(img, title=f"{t} {name}".strip(), description=desc)

def main():
    now = now_jst()
    if not FORCE_RUN and is_weekend(now):
        print(f"[SKIP] {now:%F %R} 週末のためスキップ（FORCE_RUN=1で強制実行）")
        return

    tickers = nikkei225_tickers
    raw = fetch_market_data(tickers, lookback_days=LOOKBACK_DAYS)

    if raw is None or raw.empty:
        discord_send_text(f"【ATRレンジ候補】 {now:%m/%d %H:%M}\nデータ取得失敗")
        return

    # Save metrics (all tickers)
    all_df = compute_all_metrics(raw, tickers)
    if all_df is not None and not all_df.empty:
        os.makedirs(METRICS_OUT_DIR, exist_ok=True)
        ts = now.strftime("%Y%m%d_%H%M")
        out_path = os.path.join(METRICS_OUT_DIR, f"{METRICS_PREFIX}_{ts}.csv")
        all_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] metrics csv saved: {out_path}")

    df = screen_candidates(raw, tickers)
    notify(df, raw)

if __name__ == "__main__":
    main()
