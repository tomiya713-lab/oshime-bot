# ============================================================
# ATR swing (Tableau一致・安全版・最終)
# ============================================================

import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ===== 日経225ティッカー =====
nikkei225_tickers = [
    '4151.T','4502.T','4503.T','4506.T','4507.T','4519.T','4523.T','4568.T','4578.T',
    '6479.T','6501.T','6503.T','6504.T','6506.T','6526.T','6594.T','6645.T','6674.T',
    '6701.T','6702.T','6723.T','6724.T','6752.T','6753.T','6758.T','6762.T','6770.T',
    '6841.T','6857.T','6861.T','6902.T','6920.T','6952.T','6954.T','6971.T','6976.T',
    '6981.T','7735.T','7751.T','7752.T','8035.T','7201.T','7202.T','7203.T','7205.T',
    '7211.T','7261.T','7267.T','7269.T','7270.T','7272.T','4543.T','4902.T','6146.T',
    '7731.T','7733.T','7741.T','7762.T','9432.T','9433.T','9434.T','6963.T','9984.T',
    '5831.T','7186.T','8304.T','8306.T','8308.T','8309.T','8316.T','8331.T','8354.T',
    '8411.T','8253.T','8591.T','8697.T','8601.T','8604.T','8630.T','8725.T','8750.T',
    '8766.T','8795.T','1332.T','2002.T','2269.T','2282.T','2501.T','2502.T','2503.T',
    '2801.T','2802.T','2871.T','2914.T','3086.T','3092.T','3099.T','3382.T','7453.T',
    '8233.T','8252.T','8267.T','9843.T','9983.T','2413.T','2432.T','3659.T','4307.T',
    '4324.T','4385.T','4661.T','4689.T','4704.T','4751.T','4755.T','6098.T','6178.T',
    '7974.T','9602.T','9735.T','9766.T','1605.T','3401.T','3402.T','3861.T','3405.T',
    '3407.T','4004.T','4005.T','4021.T','4042.T','4043.T','4061.T','4063.T','4183.T',
    '4188.T','4208.T','4452.T','4901.T','4911.T','6988.T','5019.T','5020.T','5101.T',
    '5108.T','5201.T','5214.T','5233.T','5301.T','5332.T','5333.T','5401.T','5406.T',
    '5411.T','3436.T','5706.T','5711.T','5713.T','5714.T','5801.T','5802.T','5803.T',
    '2768.T','8001.T','8002.T','8015.T','8031.T','8053.T','8058.T','1721.T','1801.T',
    '1802.T','1803.T','1808.T','1812.T','1925.T','1928.T','1963.T','5631.T','6103.T',
    '6113.T','6273.T','6301.T','6302.T','6305.T','6326.T','6361.T','6367.T','6471.T',
    '6472.T','6473.T','7004.T','7011.T','7013.T','7012.T','7832.T','7911.T','7912.T',
    '7951.T','3289.T','8801.T','8802.T','8804.T','8830.T','9001.T','9005.T','9007.T',
    '9008.T','9009.T','9020.T','9021.T','9022.T','9064.T','9147.T','9101.T','9104.T',
    '9107.T','9201.T','9202.T','9301.T','9501.T','9502.T','9503.T','9531.T','9532.T'
]

# ===== 短縮名マップ =====
ticker_name_map = {
    "1332.T": "日水", "1333.T": "マルハニチロ", "1605.T": "INPEX",
    "1801.T": "大成建", "1802.T": "清水建", "1803.T": "飛島建", "1808.T": "長谷工",
    "1812.T": "鹿島", "1925.T": "大和ハウス", "1928.T": "積水ハウス",
    "2002.T": "日清粉G", "2269.T": "明治HD", "2282.T": "日本ハム",
    "2413.T": "エムスリー", "2432.T": "DeNA",
    "2501.T": "サッポロHD", "2502.T": "アサヒGHD", "2503.T": "キリンHD",
    "2801.T": "キッコマン", "2802.T": "味の素", "2871.T": "ニチレイ", "2914.T": "JT",
    "3086.T": "Jフロント", "3092.T": "ZOZO", "3099.T": "三越伊勢丹",
    "3382.T": "セブン&アイ", "3401.T": "帝人", "3402.T": "東レ",
    "3405.T": "クラレ", "3407.T": "旭化成", "3436.T": "SUMCO",
    "3861.T": "王子HD", "4004.T": "昭電工", "4005.T": "住友化学",
    "4021.T": "日産化", "4042.T": "東ソー", "4043.T": "トクヤマ",
    "4061.T": "電化", "4063.T": "信越化", "4183.T": "三井化学",
    "4188.T": "三菱ケミHD", "4208.T": "UBE", "4452.T": "花王",
    "4502.T": "武田", "4503.T": "アステラス", "4506.T": "大日本住友",
    "4507.T": "塩野義", "4519.T": "中外", "4523.T": "エーザイ",
    "4543.T": "テルモ", "4568.T": "第一三共", "4578.T": "大塚HD",
    "4661.T": "OLC", "4689.T": "ZHD", "4704.T": "トレンド",
    "4751.T": "サイバー", "4755.T": "楽天G",
    "4901.T": "富士フイルム", "4902.T": "コニカミノルタ",
    "4911.T": "資生堂",
    "5101.T": "横浜ゴム", "5108.T": "ブリヂストン",
    "5201.T": "AGC", "5332.T": "TOTO", "5333.T": "日本ガイシ",
    "5401.T": "日本製鉄", "5411.T": "JFE",
    "5801.T": "古河電工", "5802.T": "住友電工", "5803.T": "フジクラ",
    "6301.T": "コマツ", "6326.T": "クボタ",
    "6367.T": "ダイキン",
    "6501.T": "日立", "6503.T": "三菱電機", "6506.T": "安川電機",
    "6758.T": "ソニーG", "6762.T": "TDK",
    "6902.T": "デンソー", "6920.T": "レーザーテック",
    "6981.T": "村田製作所",
    "7203.T": "トヨタ", "7267.T": "ホンダ",
    "7272.T": "ヤマハ発",
    "7735.T": "スクリン", "7751.T": "キヤノン",
    "7974.T": "任天堂",
    "8306.T": "三菱UFJ", "8411.T": "みずほ",
    "8591.T": "オリックス",
    "8766.T": "東京海上",
    "9432.T": "NTT", "9433.T": "KDDI", "9434.T": "SB"
}

# ===== Discord =====
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# ================= 指標計算 =================
def wilder_ema(series, period):
    alpha = 1 / period
    out = series.copy()
    out.iloc[:period] = np.nan
    for i in range(period, len(series)):
        if i == period:
            out.iloc[i] = series.iloc[:period].mean()
        else:
            out.iloc[i] = out.iloc[i-1] + alpha * (series.iloc[i] - out.iloc[i-1])
    return out

def compute_indicators(df):
    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr20 = tr.rolling(20).mean()
    atr20_pct = atr20 / close * 100

    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    bb_up1 = ma20 + sd20
    bb_dn1 = ma20 - sd20

    touch_up = (high >= bb_up1.shift(1)).astype(int)
    touch_dn = (low <= bb_dn1.shift(1)).astype(int)
    tu20 = touch_up.rolling(20).sum()
    td20 = touch_dn.rolling(20).sum()

    sma25 = close.rolling(25).mean()
    sma25_slope5 = (sma25 - sma25.shift(5)) / sma25.shift(5) * 100

    up_move = high.diff()
    dn_move = -low.diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr_w = wilder_ema(tr, 14)
    plus_di = 100 * wilder_ema(plus_dm, 14) / tr_w
    minus_di = 100 * wilder_ema(minus_dm, 14) / tr_w
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx14 = wilder_ema(dx, 14)

    return atr20_pct, tu20, td20, sma25_slope5, adx14, bb_up1, bb_dn1

# ================= メイン =================
messages = []

for t in nikkei225_tickers:
    try:
        df = yf.download(t, period="1y", interval="1d", progress=False)
        if df.empty:
            continue

        atrp, tu, td, slope, adx, bb_u, bb_d = compute_indicators(df)
        last = df.index[-1]

        if (
            pd.notna(adx.loc[last]) and adx.loc[last] <= 25 and
            pd.notna(atrp.loc[last]) and 1.8 <= atrp.loc[last] <= 4.0 and
            tu.loc[last] >= 3 and td.loc[last] >= 3 and
            abs(slope.loc[last]) <= 0.5
        ):
            name = ticker_name_map.get(t, t)
            close = df["Close"].iloc[-1]
            msg = (
                f"【ATR Swing（Tableau一致）】\n"
                f"{t} {name}\n"
                f"Close: {close:.2f}\n"
                f"BB +1σ: {bb_u.iloc[-1]:.2f} / -1σ: {bb_d.iloc[-1]:.2f}\n"
                f"ATR20%: {atrp.loc[last]:.2f}% | ADX14: {adx.loc[last]:.1f}\n"
                f"±1σタッチ(20D): +{int(tu.loc[last])} / -{int(td.loc[last])}\n"
                f"SMA25傾き(5D): {slope.loc[last]:.2f}%"
            )
            messages.append(msg)

    except Exception as e:
        print(f"Error {t}: {e}")

if messages and DISCORD_WEBHOOK_URL:
    for m in messages:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": m})
