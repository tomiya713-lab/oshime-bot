# -*- coding: utf-8 -*-
import os
import sys
import math
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

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

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ==========================
# Utility
# ==========================
def safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def post_discord(message: str):
    if not DISCORD_WEBHOOK_URL:
        print("[WARN] DISCORD_WEBHOOK_URL is empty. Skip posting.")
        return
    payload = {"content": message}
    r = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=30)
    r.raise_for_status()

# ==========================
# Data download
# ==========================
def download_prices(tickers, period="1y", interval="1d"):
    """
    yfinance: 調整後OHLCで取得する（auto_adjust=True）
    - indicators_latest_by_ticker_v2 作成時と前提を合わせるため
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    # --- ここだけ変更（auto_adjust=True） ---
    df = yf.download(
        tickers,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,   # ★重要：調整後OHLCに統一
        threads=True,
    )
    return df

# ==========================
# Indicator calculations（既存ロジックのまま）
# ==========================
def calc_sma(series, window):
    return series.rolling(window).mean()

def calc_atr_pct(high, low, close, window=20):
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    atr = tr.rolling(window).mean()
    atr_pct = (atr / close) * 100.0
    return atr_pct

def calc_adx(high, low, close, window=14):
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window).sum()

    plus_di = 100 * plus_dm.rolling(window).sum() / atr
    minus_di = 100 * minus_dm.rolling(window).sum() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window).mean()
    return adx

def calc_bb_touch_counts(close, high, low, window=20, sigma=1.0):
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    upper = ma + sigma * sd
    lower = ma - sigma * sd

    up_touch = (high >= upper).astype(int)
    dn_touch = (low <= lower).astype(int)

    up_cnt = up_touch.rolling(window).sum()
    dn_cnt = dn_touch.rolling(window).sum()
    return up_cnt, dn_cnt

def latest_atr_swing_from_raw(raw_df: pd.DataFrame, tickers: list):
    rows = []
    for ticker in tickers:
        try:
            # --- ここは元コードの形を維持しつつ、Adj Closeが無いケースだけ吸収 ---
            if isinstance(raw_df.columns, pd.MultiIndex):
                # yfinanceのauto_adjust設定により列構成が変わるため吸収
                if ("Adj Close", ticker) in raw_df.columns:
                    close = raw_df[("Adj Close", ticker)].dropna()
                else:
                    close = raw_df[("Close", ticker)].dropna()
                high = raw_df[("High", ticker)].reindex(close.index)
                low = raw_df[("Low", ticker)].reindex(close.index)
            else:
                # 単一銘柄でMultiIndexじゃない場合
                if "Adj Close" in raw_df.columns:
                    close = raw_df["Adj Close"].dropna()
                else:
                    close = raw_df["Close"].dropna()
                high = raw_df["High"].reindex(close.index)
                low = raw_df["Low"].reindex(close.index)

            if len(close) < 80:
                continue

            sma25 = calc_sma(close, 25)
            atr20_pct = calc_atr_pct(high, low, close, 20)
            adx14 = calc_adx(high, low, close, 14)

            up_cnt, dn_cnt = calc_bb_touch_counts(close, high, low, 20, sigma=1.0)

            last_dt = close.index[-1]
            last_close = safe_float(close.iloc[-1])
            last_sma25 = safe_float(sma25.iloc[-1])
            last_atr20_pct = safe_float(atr20_pct.iloc[-1])
            last_adx14 = safe_float(adx14.iloc[-1])
            last_up_cnt = safe_float(up_cnt.iloc[-1])
            last_dn_cnt = safe_float(dn_cnt.iloc[-1])

            if last_close is None or last_sma25 is None or last_atr20_pct is None or last_adx14 is None:
                continue

            sma_dev_pct = ((last_close - last_sma25) / last_sma25) * 100.0 if last_sma25 != 0 else None
            sma_dev_abs = abs(sma_dev_pct) if sma_dev_pct is not None else None

            rows.append({
                "Date": last_dt.strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Close": last_close,
                "SMA25": last_sma25,
                "SMA_dev_pct": sma_dev_pct,
                "ATR20_pct": last_atr20_pct,
                "ADX14": last_adx14,
                "BB_up_1sigma_touch_cnt20": last_up_cnt,
                "BB_dn_1sigma_touch_cnt20": last_dn_cnt,
                "SMA_dev_abs": sma_dev_abs,
            })
        except Exception as e:
            print(f"[WARN] failed ticker={ticker}: {e}")
            continue

    return pd.DataFrame(rows)

# ==========================
# Main（抽出条件・Discord通知は既存のまま）
# ==========================
def main():
    tickers = nikkei225_tickers[:]  # 既存のまま
    raw = download_prices(tickers, period="1y", interval="1d")

    latest_df = latest_atr_swing_from_raw(raw, tickers)
    if latest_df.empty:
        post_discord("【ATR Swing】本日の候補：0件")
        return

    # ---- 抽出ロジック（既存のまま） ----
    # ① ADX<=25
    # ② ±1σタッチ回数>=3（上・下それぞれ>=3）
    # ③ 1.8<=ATR<=4
    # ④ SMA25乖離率(|%|)<=0.5
    latest_df["BB_up_1sigma_touch_cnt20"] = pd.to_numeric(latest_df["BB_up_1sigma_touch_cnt20"], errors="coerce")
    latest_df["BB_dn_1sigma_touch_cnt20"] = pd.to_numeric(latest_df["BB_dn_1sigma_touch_cnt20"], errors="coerce")
    latest_df["ATR20_pct"] = pd.to_numeric(latest_df["ATR20_pct"], errors="coerce")
    latest_df["ADX14"] = pd.to_numeric(latest_df["ADX14"], errors="coerce")
    latest_df["SMA_dev_abs"] = pd.to_numeric(latest_df["SMA_dev_abs"], errors="coerce")

    filtered = latest_df[
        (latest_df["ADX14"] <= 25) &
        (latest_df["BB_up_1sigma_touch_cnt20"] >= 3) &
        (latest_df["BB_dn_1sigma_touch_cnt20"] >= 3) &
        (latest_df["ATR20_pct"] >= 1.8) & (latest_df["ATR20_pct"] <= 4.0) &
        (latest_df["SMA_dev_abs"] <= 0.5)
    ].copy()

    filtered = filtered.sort_values(["ATR20_pct"], ascending=False)

    # ---- Discord通知（既存のまま） ----
    lines = []
    lines.append(f"【ATR Swing】本日の候補：{len(filtered)}件\n")
    for _, r in filtered.iterrows():
        t = r["Ticker"]
        name = ticker_name_map.get(t, "")
        if name:
            disp = f"{t} {name}"
        else:
            disp = f"{t}"
        lines.append(
            f"- {disp} | "
            f"Close={r['Close']:.1f} "
            f"SMA25={r['SMA25']:.1f} "
            f"ATR20%={r['ATR20_pct']:.2f} "
            f"ADX14={r['ADX14']:.1f} "
            f"BBtouch(up/dn)={int(r['BB_up_1sigma_touch_cnt20'])}/{int(r['BB_dn_1sigma_touch_cnt20'])}"
        )

    post_discord("\n".join(lines))

if __name__ == "__main__":
    main()
