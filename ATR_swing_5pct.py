# -*- coding: utf-8 -*-
import os
import math
import datetime
import pandas as pd
import yfinance as yf
import requests

# ============================================================
# 重要：銘柄リスト / TickerMap は変えない（ここはそのまま）
# ============================================================

# ===== 日経225ティッカー ====3D
nikkei225_tickers = [ '4151.T','4502.T','4503.T','4506.T','4507.T','4519.T','4523.T','4568.T','4578.T','5019.T',
 '5020.T','5101.T','5108.T','5201.T','5202.T','5214.T','5232.T','5233.T','5301.T','5332.T',
 '5333.T','5401.T','5406.T','5411.T','5541.T','5631.T','5706.T','5707.T','5711.T','5713.T',
 '5714.T','5801.T','5802.T','5803.T','5901.T','6098.T','6103.T','6113.T','6178.T','6273.T',
 '6301.T','6302.T','6305.T','6326.T','6361.T','6367.T','6471.T','6472.T','6473.T','6501.T',
 '6503.T','6504.T','6506.T','6645.T','6674.T','6701.T','6702.T','6724.T','6752.T','6753.T',
 '6758.T','6762.T','6770.T','6841.T','6857.T','6861.T','6902.T','6952.T','6954.T','6963.T',
 '6971.T','6976.T','6981.T','7011.T','7012.T','7013.T','7201.T','7202.T','7203.T','7205.T',
 '7211.T','7261.T','7267.T','7269.T','7270.T','7272.T','7731.T','7733.T','7735.T','7741.T',
 '7751.T','7752.T','7762.T','7832.T','7911.T','8001.T','8002.T','8015.T','8031.T','8035.T',
 '8053.T','8058.T','8233.T','8252.T','8267.T','8306.T','8308.T','8309.T','8316.T','8331.T',
 '8354.T','8411.T','8601.T','8604.T','8628.T','8630.T','8725.T','8750.T','8766.T','8795.T',
 '8801.T','8802.T','8830.T','9001.T','9005.T','9007.T','9008.T','9009.T','9020.T','9021.T',
 '9022.T','9062.T','9064.T','9101.T','9104.T','9107.T','9201.T','9202.T','9301.T','9412.T',
 '9432.T','9433.T','9434.T','9501.T','9502.T','9503.T','9531.T','9532.T','9613.T','9983.T',
 '9984.T'
]

ticker_name_map = {
    "4151.T": "協和キリン",
    "4502.T": "武田薬品工業",
    "4503.T": "アステラス製薬",
    "4506.T": "住友ファーマ",
    "4507.T": "塩野義製薬",
    "4519.T": "中外製薬",
    "4523.T": "エーザイ",
    "4568.T": "第一三共",
    "4578.T": "大塚HD",
    "5019.T": "出光興産",
    "5020.T": "ENEOS",
    "5101.T": "横浜ゴム",
    "5108.T": "ブリヂストン",
    "5201.T": "AGC",
    "5202.T": "日本板硝子",
    "5214.T": "日本電気硝子",
    "5232.T": "住友大阪セメント",
    "5233.T": "太平洋セメント",
    "5301.T": "東海カーボン",
    "5332.T": "TOTO",
    "5333.T": "日本ガイシ",
    "5401.T": "日本製鉄",
    "5406.T": "神戸製鋼所",
    "5411.T": "JFE",
    "5541.T": "大平金",
    "5631.T": "日本製鋼所",
    "5706.T": "三井金属",
    "5707.T": "東邦亜鉛",
    "5711.T": "三菱マテリアル",
    "5713.T": "住友金属鉱山",
    "5714.T": "DOWA",
    "5801.T": "古河電工",
    "5802.T": "住友電工",
    "5803.T": "フジクラ",
    "5901.T": "東プレ",
    "6098.T": "リクルート",
    "6103.T": "オークマ",
    "6113.T": "アマダ",
    "6178.T": "日本郵政",
    "6273.T": "SMC",
    "6301.T": "コマツ",
    "6302.T": "住友重機",
    "6305.T": "日立建機",
    "6326.T": "クボタ",
    "6361.T": "荏原",
    "6367.T": "ダイキン",
    "6471.T": "日本精工",
    "6472.T": "NTN",
    "6473.T": "ジェイテクト",
    "6501.T": "日立",
    "6503.T": "三菱電機",
    "6504.T": "富士電機",
    "6506.T": "安川電機",
    "6645.T": "オムロン",
    "6674.T": "GSユアサ",
    "6701.T": "NEC",
    "6702.T": "富士通",
    "6724.T": "セイコーエプソン",
    "6752.T": "パナソニックHD",
    "6753.T": "シャープ",
    "6758.T": "ソニーG",
    "6762.T": "TDK",
    "6770.T": "アルプスアルパイン",
    "6841.T": "横河電機",
    "6857.T": "アドバンテスト",
    "6861.T": "キーエンス",
    "6902.T": "デンソー",
    "6952.T": "カシオ",
    "6954.T": "ファナック",
    "6963.T": "ローム",
    "6971.T": "京セラ",
    "6976.T": "太陽誘電",
    "6981.T": "村田製作所",
    "7011.T": "三菱重工",
    "7012.T": "川崎重工",
    "7013.T": "IHI",
    "7201.T": "日産自",
    "7202.T": "いすゞ",
    "7203.T": "トヨタ",
    "7205.T": "日野自",
    "7211.T": "三菱自",
    "7261.T": "マツダ",
    "7267.T": "ホンダ",
    "7269.T": "スズキ",
    "7270.T": "SUBARU",
    "7272.T": "ヤマハ発動機",
    "7731.T": "ニコン",
    "7733.T": "オリンパス",
    "7735.T": "SCREEN",
    "7741.T": "HOYA",
    "7751.T": "キヤノン",
    "7752.T": "リコー",
    "7762.T": "シチズン",
    "7832.T": "バンダイナムコ",
    "7911.T": "TOPPAN",
    "8001.T": "伊藤忠",
    "8002.T": "丸紅",
    "8015.T": "豊田通商",
    "8031.T": "三井物産",
    "8035.T": "東京エレク",
    "8053.T": "住友商事",
    "8058.T": "三菱商事",
    "8233.T": "高島屋",
    "8252.T": "丸井G",
    "8267.T": "イオン",
    "8306.T": "三菱UFJ",
    "8308.T": "りそなHD",
    "8309.T": "三井住友トラスト",
    "8316.T": "三井住友FG",
    "8331.T": "千葉銀行",
    "8354.T": "ふくおかFG",
    "8411.T": "みずほFG",
    "8601.T": "大和証券G",
    "8604.T": "野村HD",
    "8628.T": "松井証券",
    "8630.T": "SOMPO",
    "8725.T": "MS&AD",
    "8750.T": "第一生命",
    "8766.T": "東京海上",
    "8795.T": "T&D",
    "8801.T": "三井不動産",
    "8802.T": "三菱地所",
    "8830.T": "住友不動産",
    "9001.T": "東武",
    "9005.T": "東急",
    "9007.T": "小田急",
    "9008.T": "京王",
    "9009.T": "京成",
    "9020.T": "JR東日本",
    "9021.T": "JR西日本",
    "9022.T": "JR東海",
    "9062.T": "日本通運",
    "9064.T": "ヤマトHD",
    "9101.T": "日本郵船",
    "9104.T": "商船三井",
    "9107.T": "川崎汽船",
    "9201.T": "JAL",
    "9202.T": "ANA",
    "9301.T": "三菱倉庫",
    "9412.T": "スカパーJSAT",
    "9432.T": "NTT",
    "9433.T": "KDDI",
    "9434.T": "ソフトバンク",
    "9501.T": "東京電力HD",
    "9502.T": "中部電力",
    "9503.T": "関西電力",
    "9531.T": "東京ガス",
    "9532.T": "大阪ガス",
    "9613.T": "NTTデータ",
    "9983.T": "ファーストリテイリング",
    "9984.T": "ソフトバンクG",
}

# ============================================================
# 抽出条件（ここだけ差し替え）
# ============================================================
ADX_MAX = 25.0
ATR_MIN_PCT = 1.8
ATR_MAX_PCT = 4.0
BB_TOUCH_MIN = 3
SMA_DEV_MAX_PCT = 0.5  # |(Close-SMA25)/SMA25*100| <= 0.5

# 期間：直近60営業日（計算に必要なrollingもあるので取得自体はもう少し長め）
LOOKBACK_TRADING_DAYS = 60
DOWNLOAD_PERIOD = "6mo"  # 余裕を持ってrollingを安定させる

# ============================================================
# Discord送信（ここは触らない）
# ============================================================
def send_long_text(webhook_url: str, text: str, chunk_size: int = 1800) -> None:
    if not text:
        return
    if len(text) <= chunk_size:
        requests.post(webhook_url, json={"content": text}, timeout=20)
        return

    start = 0
    while start < len(text):
        chunk = text[start:start + chunk_size]
        requests.post(webhook_url, json={"content": chunk}, timeout=20)
        start += chunk_size


# ============================================================
# 指標計算（indicators_latest_by_ticker_v2 を作った時の定義）
#   - ATR20_pct：TRの20日単純移動平均 / Close * 100
#   - ADX14：WilderのADX（14）
#   - BB±1σ：BB期間20、σ=1、タッチは High>=上側 or Low<=下側
#   - タッチ回数：直近20営業日の回数
#   - SMA25乖離率：((Close - SMA25)/SMA25*100)
# ============================================================
def calc_adx14(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    # Wilder smoothing (smoothed sums)
    atr = pd.Series(math.nan, index=tr.index)
    plus_dm_s = pd.Series(math.nan, index=tr.index)
    minus_dm_s = pd.Series(math.nan, index=tr.index)

    if len(tr) > n:
        atr.iloc[n] = tr.iloc[1:n + 1].sum()
        plus_dm_s.iloc[n] = plus_dm.iloc[1:n + 1].sum()
        minus_dm_s.iloc[n] = minus_dm.iloc[1:n + 1].sum()

        for i in range(n + 1, len(tr)):
            atr.iloc[i] = atr.iloc[i - 1] - (atr.iloc[i - 1] / n) + tr.iloc[i]
            plus_dm_s.iloc[i] = plus_dm_s.iloc[i - 1] - (plus_dm_s.iloc[i - 1] / n) + plus_dm.iloc[i]
            minus_dm_s.iloc[i] = minus_dm_s.iloc[i - 1] - (minus_dm_s.iloc[i - 1] / n) + minus_dm.iloc[i]

    plus_di = 100.0 * (plus_dm_s / atr)
    minus_di = 100.0 * (minus_dm_s / atr)

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di))

    adx = pd.Series(math.nan, index=dx.index)
    if len(dx) > 2 * n:
        first_adx = dx.iloc[n + 1:2 * n + 1].mean()
        adx.iloc[2 * n] = first_adx
        for i in range(2 * n + 1, len(dx)):
            adx.iloc[i] = ((adx.iloc[i - 1] * (n - 1)) + dx.iloc[i]) / n

    return adx


def fetch_raw_data(tickers: list[str]) -> pd.DataFrame:
    # auto_adjust=True：indicators_latest_by_ticker_v2 と同じ“調整後OHLC”前提に合わせる
    raw = yf.download(
        tickers=tickers,
        period=DOWNLOAD_PERIOD,
        interval="1d",
        group_by="column",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return raw


def calc_latest_metrics_from_raw(raw: pd.DataFrame, ticker: str) -> dict | None:
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        if ("Close", ticker) not in raw.columns:
            return None
        close = raw["Close"][ticker].dropna()
        high = raw["High"][ticker].dropna()
        low = raw["Low"][ticker].dropna()
        vol = raw["Volume"][ticker].dropna() if ("Volume", ticker) in raw.columns else None
    else:
        # 単一ティッカーの場合
        close = raw["Close"].dropna()
        high = raw["High"].dropna()
        low = raw["Low"].dropna()
        vol = raw["Volume"].dropna() if "Volume" in raw.columns else None

    df = pd.DataFrame({"Close": close, "High": high, "Low": low})
    if vol is not None:
        df["Volume"] = vol
    df = df.dropna()

    # 直近60営業日に限定（ただしrolling計算はdf全体でOK）
    df_recent = df.tail(LOOKBACK_TRADING_DAYS)

    if len(df) < 60:
        return None

    close_s = df["Close"]
    high_s = df["High"]
    low_s = df["Low"]

    # --- SMA25 ---
    sma25 = close_s.rolling(25, min_periods=25).mean()
    if pd.isna(sma25.iloc[-1]):
        return None

    # --- SMA25乖離率（%） ---
    sma25_dev_pct = (close_s - sma25) / sma25 * 100.0
    if pd.isna(sma25_dev_pct.iloc[-1]):
        return None
    if abs(sma25_dev_pct.iloc[-1]) > SMA_DEV_MAX_PCT:
        return None

    # --- ATR20%（TRの20日SMA / Close * 100） ---
    prev_close = close_s.shift(1)
    tr = pd.concat(
        [(high_s - low_s).abs(), (high_s - prev_close).abs(), (low_s - prev_close).abs()],
        axis=1
    ).max(axis=1)
    atr20 = tr.rolling(20, min_periods=20).mean()
    if pd.isna(atr20.iloc[-1]):
        return None

    atr20_pct = (atr20 / close_s) * 100.0
    if pd.isna(atr20_pct.iloc[-1]):
        return None
    if not (ATR_MIN_PCT <= atr20_pct.iloc[-1] <= ATR_MAX_PCT):
        return None

    # --- ADX14（Wilder） ---
    adx14 = calc_adx14(high_s, low_s, close_s, 14)
    if pd.isna(adx14.iloc[-1]):
        return None
    if adx14.iloc[-1] > ADX_MAX:
        return None

    # --- BB(20, 1σ) とタッチ回数(20) ---
    bb_mid = close_s.rolling(20, min_periods=20).mean()
    bb_std = close_s.rolling(20, min_periods=20).std()  # ddof=1（pandasデフォルト）
    bb_up_1 = bb_mid + bb_std
    bb_dn_1 = bb_mid - bb_std

    bb_up_touch = (high_s >= bb_up_1).rolling(20, min_periods=20).sum()
    bb_dn_touch = (low_s <= bb_dn_1).rolling(20, min_periods=20).sum()

    if pd.isna(bb_up_touch.iloc[-1]) or pd.isna(bb_dn_touch.iloc[-1]):
        return None
    if (bb_up_touch.iloc[-1] < BB_TOUCH_MIN) or (bb_dn_touch.iloc[-1] < BB_TOUCH_MIN):
        return None

    # Latest values
    metrics = {
        "Ticker": ticker,
        "Date": df_recent.index[-1].strftime("%Y-%m-%d"),
        "Close": float(close_s.iloc[-1]),
        "SMA25": float(sma25.iloc[-1]),
        "SMA25_dev_pct": float(sma25_dev_pct.iloc[-1]),
        "ATR20_pct": float(atr20_pct.iloc[-1]),
        "ADX14": float(adx14.iloc[-1]),
        "BB_up_1sigma_touch_cnt20": int(bb_up_touch.iloc[-1]),
        "BB_dn_1sigma_touch_cnt20": int(bb_dn_touch.iloc[-1]),
    }
    return metrics


def screen_candidates(raw: pd.DataFrame, tickers: list[str]) -> list[dict]:
    out = []
    for t in tickers:
        m = calc_latest_metrics_from_raw(raw, t)
        if m is not None:
            out.append(m)
    # 参考通知の並び順に合わせて ATR20% 降順
    out.sort(key=lambda x: x.get("ATR20_pct", 0), reverse=True)
    return out


def notify(candidates: list[dict], timestamp_jst: str) -> None:
    """Discordに通知（抽出ロジック以外は変更しない想定）"""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print("DISCORD_WEBHOOK_URL is empty. Skip notify.")
        return

    if not candidates:
        send_long_text(webhook_url, f"【ATR Swing】本日の候補：0件\n\n{timestamp_jst}")
        return

    # 参考の通知文に寄せる（候補数＋箇条書き）
    lines = [f"【ATR Swing】本日の候補：{len(candidates)}件", ""]
    for c in candidates:
        name = ticker_name_map.get(c["Ticker"], "")
        head = f"- {c['Ticker']} {name}" if name else f"- {c['Ticker']}"
        lines.append(
            f"{head} | Close={c['Close']:.1f} SMA25={c['SMA25']:.1f} "
            f"ATR20%={c['ATR20_pct']:.2f} ADX14={c['ADX14']:.1f} "
            f"BBtouch(up/dn)={int(c['BB_up_1sigma_touch_cnt20'])}/{int(c['BB_dn_1sigma_touch_cnt20'])}"
        )

    lines += ["", timestamp_jst]
    msg = "\n".join(lines)
    send_long_text(webhook_url, msg)


def main() -> None:
    # JST timestamp
    jst = datetime.timezone(datetime.timedelta(hours=9))
    now_jst = datetime.datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S JST")

    raw = fetch_raw_data(nikkei225_tickers)
    candidates = screen_candidates(raw, nikkei225_tickers)

    notify(candidates, now_jst)


if __name__ == "__main__":
    main()
