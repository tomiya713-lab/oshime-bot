import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ====== チャート用（元の運用に合わせて：入っていれば画像も送る） ======
try:
    import mplfinance as mpf
    MPF_AVAILABLE = True
except Exception:
    MPF_AVAILABLE = False


# =========================
# Discord
# =========================
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

def post_discord(message: str):
    if not DISCORD_WEBHOOK_URL:
        print("[WARN] DISCORD_WEBHOOK_URL is empty. Skip posting.")
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=20)
    except Exception as e:
        print(f"[WARN] Discord post failed: {e}")


# =========================
# ここから “既存のリスト/マップは変更しない”
# =========================

# ===== 日経225ティッカー ====3D
nikkei225_tickers = [ '4151.T','4502.T','4503.T','4506.T','4507.T','4519.T','4523.T','4543.T','4568.T','4578.T','4901.T','4902.T','4911.T','4911.T','5019.T','5020.T','5101.T','5108.T','5201.T','5214.T','5232.T','5233.T','5301.T','5332.T','5333.T','5401.T','5406.T','5411.T','5631.T','5706.T','5711.T','5713.T','5714.T','5801.T','5802.T','5803.T','5831.T','6098.T','6103.T','6113.T','6146.T','6178.T','6301.T','6302.T','6305.T','6326.T','6361.T','6367.T','6471.T','6472.T','6473.T','6479.T','6501.T','6503.T','6504.T','6506.T','6526.T','6594.T','6645.T','6674.T','6701.T','6702.T','6723.T','6724.T','6752.T','6753.T','6758.T','6762.T','6770.T','6841.T','6857.T','6861.T','6902.T','6920.T','6952.T','6954.T','6963.T','6971.T','6976.T','6981.T','6988.T','7004.T','7011.T','7012.T','7013.T','7186.T','7201.T','7202.T','7203.T','7205.T','7211.T','7261.T','7267.T','7269.T','7270.T','7272.T','7453.T','7731.T','7733.T','7735.T','7741.T','7751.T','7752.T','7762.T','7832.T','7911.T','7912.T','7951.T','7974.T','8001.T','8002.T','8015.T','8031.T','8035.T','8053.T','8058.T','8233.T','8252.T','8253.T','8267.T','8304.T','8306.T','8308.T','8309.T','8316.T','8331.T','8354.T','8411.T','8591.T','8601.T','8604.T','8630.T','8697.T','8725.T','8750.T','8766.T','8795.T','8801.T','8802.T','8804.T','8830.T','9001.T','9005.T','9007.T','9008.T','9009.T','9020.T','9021.T','9022.T','9064.T','9101.T','9104.T','9107.T','9147.T','9201.T','9202.T','9301.T','9432.T','9433.T','9434.T','9501.T','9502.T','9503.T','9531.T','9532.T' ]

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
    "4568.T": "第一三共", "4578.T": "大塚HD", "4901.T": "富士フイルム", "4902.T": "コニカミノルタ",
    "4911.T": "資生堂", "4911.T": "資生堂", "5019.T": "出光", "5020.T": "ENEOS",
    "5101.T": "浜ゴム", "5108.T": "ブリヂストン", "5201.T": "AGC", "5214.T": "日電硝",
    "5232.T": "住友大阪", "5233.T": "太平洋セメ", "5301.T": "東海カ", "5332.T": "TOTO",
    "5333.T": "ガイシ", "5401.T": "日本製鉄", "5406.T": "神戸鋼", "5411.T": "JFE",
    "5631.T": "日製鋼", "5706.T": "三井金", "5711.T": "三菱マ", "5713.T": "住友鉱",
    "5714.T": "DOWA", "5801.T": "古河電", "5802.T": "住友電", "5803.T": "フジクラ",
    "5831.T": "しずおかFG", "6098.T": "リクルート", "6103.T": "オークマ", "6113.T": "アマダ",
    "6146.T": "ディスコ", "6178.T": "日本郵政", "6301.T": "コマツ", "6302.T": "住友重",
    "6305.T": "日立建機", "6326.T": "クボタ", "6361.T": "荏原", "6367.T": "ダイキン",
    "6471.T": "日精工", "6472.T": "NTN", "6473.T": "ジェイテクト", "6479.T": "ミネベア",
    "6501.T": "日立", "6503.T": "三菱電", "6504.T": "富士電機", "6506.T": "安川電",
    "6526.T": "ソシオネクスト", "6594.T": "日本電産", "6645.T": "オムロン", "6674.T": "GSユアサ",
    "6701.T": "NEC", "6702.T": "富士通", "6723.T": "ルネサス", "6724.T": "セイコーEP",
    "6752.T": "パナソニック", "6753.T": "シャープ", "6758.T": "ソニーG", "6762.T": "TDK",
    "6770.T": "アルプスアル", "6841.T": "横河電", "6857.T": "アドバンテスト", "6861.T": "キーエンス",
    "6902.T": "デンソー", "6920.T": "レーザーテック", "6952.T": "カシオ", "6954.T": "ファナック",
    "6963.T": "ローム", "6971.T": "京セラ", "6976.T": "太陽誘電", "6981.T": "村田製",
    "6988.T": "日東電", "7004.T": "日立造", "7011.T": "三菱重", "7012.T": "川重",
    "7013.T": "IHI", "7186.T": "コンコルディア", "7201.T": "日産", "7202.T": "いすゞ",
    "7203.T": "トヨタ", "7205.T": "日野自", "7211.T": "三菱自", "7261.T": "マツダ",
    "7267.T": "ホンダ", "7269.T": "スズキ", "7270.T": "SUBARU", "7272.T": "ヤマハ発",
    "7453.T": "良品計画", "7731.T": "ニコン", "7733.T": "オリンパス", "7735.T": "SCREEN",
    "7741.T": "HOYA", "7751.T": "キヤノン", "7752.T": "リコー", "7762.T": "シチズン",
    "7832.T": "バンナムHD", "7911.T": "凸版", "7912.T": "大日印", "7951.T": "ヤマハ",
    "7974.T": "任天堂", "8001.T": "伊藤忠", "8002.T": "丸紅", "8015.T": "豊田通商",
    "8031.T": "三井物", "8035.T": "東エレ", "8053.T": "住友商", "8058.T": "三菱商",
    "8233.T": "高島屋", "8252.T": "丸井G", "8253.T": "クレセゾン", "8267.T": "イオン",
    "8304.T": "あおぞら", "8306.T": "三菱UFJ", "8308.T": "りそなHD", "8309.T": "三井住友トラ",
    "8316.T": "三井住友FG", "8331.T": "千葉銀", "8354.T": "ふくおかFG", "8411.T": "みずほFG",
    "8591.T": "オリックス", "8601.T": "大和証G", "8604.T": "野村HD", "8630.T": "SOMPO",
    "8697.T": "JPX", "8725.T": "MS&AD", "8750.T": "第一生命", "8766.T": "東京海上",
    "8795.T": "T&D", "8801.T": "三井不", "8802.T": "菱地所", "8804.T": "東建物",
    "8830.T": "住友不", "9001.T": "東武", "9005.T": "東急", "9007.T": "小田急",
    "9008.T": "京王", "9009.T": "京成", "9020.T": "JR東", "9021.T": "JR西",
    "9022.T": "JR東海", "9064.T": "ヤマトHD", "9101.T": "郵船", "9104.T": "商船三井",
    "9107.T": "川崎汽", "9147.T": "NXHD", "9201.T": "JAL", "9202.T": "ANA",
    "9301.T": "三菱倉", "9432.T": "NTT", "9433.T": "KDDI", "9434.T": "SBG通信",
    "9501.T": "東電HD", "9502.T": "中部電", "9503.T": "関西電", "9531.T": "東ガス",
    "9532.T": "大ガス",
}
# =========================
# ここまで “既存のリスト/マップは変更しない”
# =========================


# =========================
# ここから “エラー修正（MultiIndex対策）”
# =========================
def _normalize_downloaded_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    yfinance が MultiIndex 列で返してきた場合でも、
    単一銘柄の OHLCV 形式に正規化して返す。
    """
    if df is None or df.empty:
        return df

    # MultiIndex columns: 例 [('Close','8308.T'), ...] や [('8308.T','Close'), ...]
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)

        if ticker in lv0:
            # df[ticker] が OHLCV になるパターン
            df = df[ticker].copy()
        elif ticker in lv1:
            # level=1 側に ticker があるパターン
            df = df.xs(ticker, axis=1, level=1).copy()
        else:
            # 最後の手段：列名をフラット化
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # 列名ゆれ吸収
    rename_map = {
        "Adj Close": "Adj Close",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    }
    # 必要な列だけ残す（存在しない列は落ちないように）
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].copy()

    # 数値化
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # index
    df = df.dropna(subset=[c for c in ["Close"] if c in df.columns])
    return df


def download_one_ticker(ticker):
    """
    - ticker が list/tuple で来ても文字列に直す
    - yfinance の返り値が MultiIndex でも正規化
    - auto_adjust=False を明示して Close を生値のまま固定
    """
    # list/tuple 対策（これが一番ありがちな原因）
    if isinstance(ticker, (list, tuple, set)):
        ticker = list(ticker)[0]

    try:
        df = yf.download(
            ticker,
            period="1y",
            interval="1d",
            progress=False,
            auto_adjust=False,     # ★ここ重要：Close を生値に固定（Adj Close も取れる）
            group_by="column",
            actions=False,
            threads=False,
        )
    except Exception as e:
        raise RuntimeError(f"download failed: {ticker} ({e})")

    df = _normalize_downloaded_df(df, ticker)
    return df


# =========================
# 指標計算（既存ロジックを壊さない範囲で）
# =========================
def wilder_rma(series: pd.Series, n: int) -> pd.Series:
    # Wilder の RMA = EMA(alpha=1/n) と同等
    return series.ewm(alpha=1/n, adjust=False).mean()


def calc_adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    # TR
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # +DM / -DM
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_rma = wilder_rma(tr, n)
    plus_dm_rma = wilder_rma(pd.Series(plus_dm, index=high.index), n)
    minus_dm_rma = wilder_rma(pd.Series(minus_dm, index=high.index), n)

    plus_di = 100 * (plus_dm_rma / tr_rma)
    minus_di = 100 * (minus_dm_rma / tr_rma)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = wilder_rma(dx, n)
    return adx


def calc_atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 20) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = wilder_rma(tr, n)
    return atr


def calc_latest_metrics_from_raw(df: pd.DataFrame):
    """
    df: 単一銘柄の OHLCV（Open/High/Low/Close/Adj Close/Volume）
    返すもの：
      - Close（生値）
      - SMA25
      - ATR20_pct（%）
      - ADX14
      - BB_up_1sigma_touch_cnt20 / BB_dn_1sigma_touch_cnt20（直近20）
    """
    need_cols = {"Open", "High", "Low", "Close"}
    if not need_cols.issubset(set(df.columns)):
        # MultiIndex 対策が効いてない、またはデータ不備
        return None

    df = df.copy().dropna()

    # 生Close固定（表示用）
    close_raw = df["Close"].astype(float)

    # 分割等の調整が必要なら Adj Close を使って “価格系列” を揃える
    # Adj Close が無い銘柄は Close で代替
    if "Adj Close" in df.columns and df["Adj Close"].notna().any():
        close_base = df["Adj Close"].astype(float)
        # 調整係数（Close=0 対策）
        denom = close_raw.replace(0, np.nan)
        adj_factor = (close_base / denom).fillna(1.0)

        high_base = (df["High"].astype(float) * adj_factor)
        low_base = (df["Low"].astype(float) * adj_factor)
    else:
        close_base = close_raw
        high_base = df["High"].astype(float)
        low_base = df["Low"].astype(float)

    # SMA25（ここは CSV 側に合わせるなら “基準価格(close_base)” で作るのが安全）
    sma25 = close_base.rolling(25).mean()

    # ATR20%（Wilder）
    atr20 = calc_atr_wilder(high_base, low_base, close_base, n=20)
    atr20_pct = (atr20 / close_base) * 100

    # ADX14（Wilder）
    adx14 = calc_adx_wilder(high_base, low_base, close_base, n=14)

    # BB(1σ) & タッチ回数（直近20、HighでもLowでもOK）
    ma20 = close_base.rolling(20).mean()
    sd20 = close_base.rolling(20).std(ddof=0)
    bb_up_1s = ma20 + sd20
    bb_dn_1s = ma20 - sd20

    # タッチ判定：High >= 上1σ, Low <= 下1σ
    up_touch = (high_base >= bb_up_1s).astype(int)
    dn_touch = (low_base <= bb_dn_1s).astype(int)

    # 直近20営業日
    up_cnt20 = int(up_touch.tail(20).sum())
    dn_cnt20 = int(dn_touch.tail(20).sum())

    latest = {
        "Close": float(close_raw.iloc[-1]),
        "SMA25": float(sma25.iloc[-1]) if not np.isnan(sma25.iloc[-1]) else np.nan,
        "ATR20_pct": float(atr20_pct.iloc[-1]) if not np.isnan(atr20_pct.iloc[-1]) else np.nan,
        "ADX14": float(adx14.iloc[-1]) if not np.isnan(adx14.iloc[-1]) else np.nan,
        "BB_up_1sigma_touch_cnt20": up_cnt20,
        "BB_dn_1sigma_touch_cnt20": dn_cnt20,
    }
    return latest


def format_line(ticker: str, m: dict):
    name = ticker_name_map.get(ticker, "")
    close = m["Close"]
    sma = m["SMA25"]
    atrp = m["ATR20_pct"]
    adx = m["ADX14"]
    up = m["BB_up_1sigma_touch_cnt20"]
    dn = m["BB_dn_1sigma_touch_cnt20"]
    return f"- {ticker} {name} | Close={close:.1f} SMA25={sma:.1f} ATR20%={atrp:.2f} ADX14={adx:.1f} BBtouch(up/dn)={up}/{dn}"


def main():
    # （既存運用を壊さない：ここでは抽出条件を変えない）
    # ここは “あなたの main_rsi.py 側で抽出して通知” の想定でも、
    # このファイル単体実行でも動くようにしてあります。
    candidates = []

    for t in nikkei225_tickers:
        try:
            df = download_one_ticker(t)
            if df is None or df.empty:
                continue

            m = calc_latest_metrics_from_raw(df)
            if m is None:
                continue

            # 既存の抽出条件がここにある場合は “ここは触らない”
            # （本ファイルではとりあえず全件収集しておく）
            candidates.append((t, m))

        except Exception as e:
            print(f"Error {t}: {e}")

    # 通知（既存整形を崩さない）
    lines = [format_line(t, m) for t, m in candidates]
    msg = f"【ATR Swing】本日の候補：{len(lines)}件\n\n" + "\n".join(lines) if lines else "【ATR Swing】本日の候補：0件"
    print(msg)
    post_discord(msg)


if __name__ == "__main__":
    main()
