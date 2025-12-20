# -*- coding: utf-8 -*-
import os
import io
import time
import json
import math
import base64
import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ====== チャート用（元の運用に合わせて：入っていれば画像も送る） ======
import matplotlib.pyplot as plt


# ===== 日経225ティッカー ====3D
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
    "9531.T": "東ガス", "9532.T": "大阪ガス", "9602.T": "東宝", "6963.T": "ローム",
    "9735.T": "セコム", "9766.T": "コナミG", "9843.T": "ニトリHD", "9983.T": "ファーストリテ",
    "9984.T": "ソフトバンクG",
}
# ===== パラメータ（抽出ロジックのみ差し替え）=====
ATR_MIN_PCT = 1.8
ATR_MAX_PCT = 4.0
ADX_MAX = 25
BB_TOUCH_MIN = 3

# ※元コードの変数名を変えない（「余計なところを触らない」ため）
#   ただし意味は「SMA25乖離率(%)の絶対値の上限」として使う
SMA_SLOPE_MAX_PCT = 0.5

# ===== Discord設定 =====
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# ===== 出力設定 =====
TOP_N = 30
SLEEP_SEC = 1.0

# ===== ユーティリティ =====
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def short_name(ticker: str) -> str:
    return ticker_name_map.get(ticker, ticker)


def send_discord_message(text: str):
    if not DISCORD_WEBHOOK_URL:
        print("[WARN] DISCORD_WEBHOOK_URL が未設定です。通知をスキップします。")
        return
    payload = {"content": text}
    r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=30)
    if r.status_code >= 300:
        print("[WARN] Discord送信失敗:", r.status_code, r.text)


def send_discord_file(filename: str, data: bytes, content: str = ""):
    if not DISCORD_WEBHOOK_URL:
        print("[WARN] DISCORD_WEBHOOK_URL が未設定です。ファイル送信をスキップします。")
        return
    files = {"file": (filename, data)}
    payload = {"content": content} if content else {}
    r = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files, timeout=60)
    if r.status_code >= 300:
        print("[WARN] Discordファイル送信失敗:", r.status_code, r.text)


def to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fetch_ohlc_from_yfinance(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """
    元運用のyfinance取得を踏襲。
    ※株式分割などを考慮して、指標計算では Adj Close 比率で OHLC を内部調整する。
      そのため、ここは auto_adjust=False（生値）で固定する。
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        # Date index -> column
        df = df.reset_index()
        return df
    except Exception as e:
        print(f"[WARN] fetch failed: {ticker} {e}")
        return None


def calc_latest_metrics_from_raw(raw_df: pd.DataFrame, ticker: str):
    """
    最新日で以下を算出し、条件を満たす場合に返す（満たさない場合はNone）

      - ADX14 <= 25
      - 1.8% <= ATR20% <= 4.0%
      - 直近20日で BB(20,±1σ) +1σ/-1σ タッチ回数 >= 3（※上側・下側それぞれカウント）
      - |SMA25との乖離率(%)| <= 0.5

    ※データは yfinance の「生値」(auto_adjust=False) を取得し、
      株式分割などを考慮するために Adj Close 比率で OHLC を調整して指標を算出します。
      （通知表示の Close は生値を表示）
    """
    if raw_df is None or len(raw_df) < 60:
        return None

    df = raw_df.copy()

    # --- 列名ゆらぎ吸収（yfinanceは大小混在しやすい）---
    colmap = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n.lower() in colmap:
                return colmap[n.lower()]
        return None

    c_close = pick("Close")
    c_high = pick("High")
    c_low = pick("Low")
    c_adj = pick("Adj Close", "Adj_Close", "AdjClose")

    if c_close is None or c_high is None or c_low is None:
        return None

    close_raw = df[c_close].astype(float)
    high_raw = df[c_high].astype(float)
    low_raw = df[c_low].astype(float)

    # --- 分割等を反映した調整OHLCを作る（Adj Closeがある時だけ）---
    if c_adj is not None:
        adj_close = df[c_adj].astype(float)
        # close_rawが0のケースを避ける
        factor = (adj_close / close_raw.replace(0, np.nan)).fillna(1.0)
        close = adj_close
        high = high_raw * factor
        low = low_raw * factor
    else:
        close = close_raw
        high = high_raw
        low = low_raw

    # =================================================================================
    # 指標計算（CSV作成時のロジック：Wilder/RMAベース）
    # =================================================================================
    def rma(x: pd.Series, n: int) -> pd.Series:
        """Wilder's RMA: 初期値は最初のn本の単純平均、その後は再帰で平滑化。"""
        x = x.astype(float)
        out = pd.Series(index=x.index, dtype=float)
        if len(x) < n:
            return out

        first = x.iloc[:n].mean()
        out.iloc[:n] = np.nan
        out.iloc[n - 1] = first
        alpha = 1.0 / n
        prev = first
        for i in range(n, len(x)):
            prev = prev + alpha * (x.iloc[i] - prev)
            out.iloc[i] = prev
        return out

    # --- ATR20% ---
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr20 = rma(tr, 20)
    atr20_pct = (atr20 / close) * 100.0

    # --- ADX14 (Wilder) ---
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index
    )

    tr14 = rma(tr, 14)
    plus_dm14 = rma(plus_dm, 14)
    minus_dm14 = rma(minus_dm, 14)

    plus_di14 = 100.0 * (plus_dm14 / tr14)
    minus_di14 = 100.0 * (minus_dm14 / tr14)

    dx = 100.0 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14)
    adx14 = rma(dx, 14)

    # --- SMA25 と乖離率(%) ---
    sma25 = close.rolling(25).mean()
    sma_dev_pct = (close / sma25 - 1.0) * 100.0

    # --- BB(20, ±1σ) タッチ回数（High/Lowで判定）---
    bb_ma20 = close.rolling(20).mean()
    bb_sd20 = close.rolling(20).std(ddof=0)
    bb_up_1s = bb_ma20 + bb_sd20
    bb_dn_1s = bb_ma20 - bb_sd20

    up_touch = (high >= bb_up_1s).astype(int)
    dn_touch = (low <= bb_dn_1s).astype(int)

    bb_up_1sigma_touch_cnt20 = up_touch.rolling(20).sum()
    bb_dn_1sigma_touch_cnt20 = dn_touch.rolling(20).sum()

    # =================================================================================
    # 最新値取り出し＆条件判定
    # =================================================================================
    close_v = float(close_raw.iloc[-1])  # 通知表示は「生値」
    sma25_v = float(sma25.iloc[-1])
    atr_pct_v = float(atr20_pct.iloc[-1])
    adx_v = float(adx14.iloc[-1])
    up_cnt_v = (
        int(bb_up_1sigma_touch_cnt20.iloc[-1])
        if not np.isnan(bb_up_1sigma_touch_cnt20.iloc[-1])
        else 0
    )
    dn_cnt_v = (
        int(bb_dn_1sigma_touch_cnt20.iloc[-1])
        if not np.isnan(bb_dn_1sigma_touch_cnt20.iloc[-1])
        else 0
    )
    sma_dev_v = float(sma_dev_pct.iloc[-1])

    # NaNガード
    if any(np.isnan(x) for x in [sma25_v, atr_pct_v, adx_v, sma_dev_v]):
        return None

    # --- 条件判定（ユーザー指定）---
    if not (adx_v <= ADX_MAX):
        return None
    if not (ATR_MIN_PCT <= atr_pct_v <= ATR_MAX_PCT):
        return None
    # ※「±1σタッチ回数>=3」は CSV ロジックに合わせて「上側・下側それぞれ>=3」
    if not (up_cnt_v >= BB_TOUCH_MIN and dn_cnt_v >= BB_TOUCH_MIN):
        return None
    # ※SMA25との乖離率(%) の絶対値 <= 0.5
    if not (abs(sma_dev_v) <= SMA_SLOPE_MAX_PCT):
        return None

    return {
        "Ticker": ticker,
        "Close": close_v,
        "SMA25": sma25_v,
        "ATR20_pct": atr_pct_v,
        "ADX14": adx_v,
        "BB_up_1sigma_touch_cnt20": up_cnt_v,
        "BB_dn_1sigma_touch_cnt20": dn_cnt_v,
        "SMA_dev_pct": sma_dev_v,
    }


def format_line(row: dict) -> str:
    t = row["Ticker"]
    name = short_name(t)
    return (
        f"- {t} {name} | "
        f"Close={row['Close']:.1f} "
        f"SMA25={row['SMA25']:.1f} "
        f"ATR20%={row['ATR20_pct']:.2f} "
        f"ADX14={row['ADX14']:.1f} "
        f"BBtouch(up/dn)={row['BB_up_1sigma_touch_cnt20']}/{row['BB_dn_1sigma_touch_cnt20']}"
    )


def main():
    candidates = []
    for ticker in nikkei225_tickers:
        df = fetch_ohlc_from_yfinance(ticker, period="1y")
        if df is None:
            continue
        m = calc_latest_metrics_from_raw(df, ticker)
        if m is not None:
            candidates.append(m)
        time.sleep(SLEEP_SEC)

    # 並び順：元の雰囲気を崩さず、ATR降順に寄せる（必要ならここは元コードに合わせてください）
    candidates = sorted(candidates, key=lambda x: x["ATR20_pct"], reverse=True)[:TOP_N]

    header = f"【ATR Swing】本日の候補：{len(candidates)}件"
    lines = [header, ""]
    lines += [format_line(r) for r in candidates]

    send_discord_message("\n".join(lines))


if __name__ == "__main__":
    main()
