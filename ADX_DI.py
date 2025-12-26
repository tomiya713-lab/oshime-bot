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


# ====== 基本設定（通知・環境変数などは変更しない想定） =====
TZ_OFFSET = 9  # JST
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "180"))
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
FORCE_RUN = os.getenv("FORCE_RUN", "0") == "1"

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

# ===== ユーティリティ（既存運用のまま） =====
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


def discord_post(payload: dict, files=None):
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


def discord_send_text(content: str):
    return discord_post({"content": content})


def discord_send_image_file(file_path: str, title: str = "", description: str = ""):
    if not os.path.exists(file_path):
        return False
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        payload = {"content": f"**{title}**\n{description}".strip()}
        return discord_post(payload, files=files)


def fp(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


# ===== データ取得（既存運用のまま） =====
def fetch_market_data(tickers, lookback_days=LOOKBACK_DAYS):
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


# =====================================================================================
# ここから下：抽出ロジック＆通知文言のみ変更（それ以外は触らない）
# =====================================================================================

# ★ 抽出ロジック用パラメータ
ADX_MAX = 25.0
ATR_MIN_PCT = 1.8
ATR_MAX_PCT = 4.0
BB_TOUCH_MIN = 3
SMA_SLOPE_MAX_PCT = 0.5

# ★ DI差分フィルタ（|+DI14 - -DI14| が小さいほど「方向感が薄い」）
DI_DIFF_MAX = float(os.getenv("DI_DIFF_MAX", "7.0"))

# ★ DI比率（+DI14 ÷ -DI14 が 3%以上 = 1.03以上）
DI_RATIO_MIN = float(os.getenv("DI_RATIO_MIN", "1.03"))

# ★ 指標CSV出力（GitHub ActionsでArtifacts化する前提でもOK）
METRICS_OUT_DIR = os.getenv("METRICS_OUT_DIR", "reports")
METRICS_PREFIX = os.getenv("METRICS_PREFIX", "atr_swing_metrics")

# チャート設定（既存のまま使う想定。未設定ならデフォルトで安全に動作）
CHART_OUT_DIR = os.getenv("CHART_OUT_DIR", "charts")
CHART_LOOKBACK_DAYS = int(os.getenv("CHART_LOOKBACK_DAYS", "90"))
CHART_TOP_N = int(os.getenv("CHART_TOP_N", "8"))


def calc_latest_metrics_from_raw(raw_df: pd.DataFrame, ticker: str):
    """
    最新日で以下を算出し、条件を満たす場合に返す（満たさない場合はNone）
      - ADX14 <= 25
      - 1.8% <= ATR20% <= 4.0%
      - 直近20日で BB(20,±1σ) +1σ/-1σ タッチ回数 >= 3
      - |SMA25の20日傾き(%)| <= 0.5
      - |+DI14 - -DI14| <= DI_DIFF_MAX
      - (+DI14 / -DI14) >= DI_RATIO_MIN

    通知追加用
      - Close（最新終値）
      - BB(20,±1σ)の価格（最新日の上下1σ）
      - Bottom_Rise_Ratio = BB_up_1 / BB_dn_1（価格レンジ比率）
    """
    m = calc_latest_metrics_all_from_raw(raw_df, ticker)
    if m is None:
        return None

    if not bool(m.get("Pass", False)):
        return None

    return {
        "Ticker": m["Ticker"],
        "Name": m["Name"],
        "Close": m["Close"],
        "ATR20_pct": m["ATR20_pct"],
        "ADX14": m["ADX14"],
        "BB_up_touch_cnt20": m["BB_up_touch_cnt20"],
        "BB_dn_touch_cnt20": m["BB_dn_touch_cnt20"],
        "BB_up_1": m["BB_up_1"],
        "BB_dn_1": m["BB_dn_1"],
        "SMA25_slope20_pct": m["SMA25_slope20_pct"],
        "PLUS_DI14": m["PLUS_DI14"],
        "MINUS_DI14": m["MINUS_DI14"],
        "DI_diff": m["DI_diff"],
        "DI_Ratio": m["DI_Ratio"],
        "Bottom_Rise_Ratio": m["Bottom_Rise_Ratio"],  # BB_up_1 / BB_dn_1
    }


def calc_latest_metrics_all_from_raw(raw_df: pd.DataFrame, ticker: str):
    """
    ★全銘柄CSV用：指標を計算できたら必ず返す（抽出条件で落とさない）
    - Pass/Fail と、各条件の判定列も返す（閾値調整用）
    """
    try:
        if not isinstance(raw_df.columns, pd.MultiIndex):
            return None

        close = raw_df[("Close", ticker)].dropna()
        high = raw_df[("High", ticker)].reindex(close.index)
        low = raw_df[("Low", ticker)].reindex(close.index)

        if len(close) < 60:
            return {
                "Ticker": ticker,
                "Name": ticker_name_map.get(ticker, ""),
                "Close": np.nan,
                "ATR20_pct": np.nan,
                "ADX14": np.nan,
                "BB_up_touch_cnt20": np.nan,
                "BB_dn_touch_cnt20": np.nan,
                "BB_up_1": np.nan,
                "BB_dn_1": np.nan,
                "SMA25_slope20_pct": np.nan,
                "PLUS_DI14": np.nan,
                "MINUS_DI14": np.nan,
                "DI_diff": np.nan,
                "DI_Ratio": np.nan,
                "Bottom_Rise_Ratio": np.nan,
                "Pass_ADX": False,
                "Pass_ATR": False,
                "Pass_BB": False,
                "Pass_SMA_Slope": False,
                "Pass_DI_Diff": False,
                "Pass_DI_Ratio": False,
                "Pass": False,
                "Reason": "insufficient_data",
            }

        # --- ATR20% ---
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1
        ).max(axis=1)
        atr20 = tr.rolling(20, min_periods=20).mean()
        atr20_pct = atr20 / close * 100.0

        # --- ADX14（Rolling Sum DI / Rolling Mean DX） ---
        prev_high = high.shift(1)
        prev_low = low.shift(1)

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)

        tr14 = tr.rolling(14, min_periods=14).sum()
        plus_dm14 = plus_dm.rolling(14, min_periods=14).sum()
        minus_dm14 = minus_dm.rolling(14, min_periods=14).sum()

        plus_di14 = 100.0 * (plus_dm14 / tr14)
        minus_di14 = 100.0 * (minus_dm14 / tr14)

        dx = 100.0 * (plus_di14 - minus_di14).abs() / (plus_di14 + minus_di14)
        adx14 = dx.rolling(14, min_periods=14).mean()

        # DI差分
        di_diff = (plus_di14 - minus_di14).abs()

        # DI比率（+DI14 ÷ -DI14）※ゼロ除算回避
        di_ratio = plus_di14 / minus_di14.replace(0, np.nan)

        # --- BB(20,±1σ) & タッチ回数（直近20日） ---
        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_std = close.rolling(20, min_periods=20).std()
        bb_up_1 = bb_mid + bb_std
        bb_dn_1 = bb_mid - bb_std

        touch_up_1 = (high >= bb_up_1).astype(int)
        touch_dn_1 = (low <= bb_dn_1).astype(int)

        touch_up_cnt20 = touch_up_1.rolling(20, min_periods=20).sum()
        touch_dn_cnt20 = touch_dn_1.rolling(20, min_periods=20).sum()

        # --- SMA25傾き（20日） ---
        sma25 = close.rolling(25, min_periods=25).mean()
        sma25_20ago = sma25.shift(20)
        sma25_slope20_pct = (sma25 - sma25_20ago) / sma25_20ago * 100.0

        # --- 最新値 ---
        close_v = close.iloc[-1]
        atr_v = atr20_pct.iloc[-1]
        adx_v = adx14.iloc[-1]
        up_cnt = touch_up_cnt20.iloc[-1]
        dn_cnt = touch_dn_cnt20.iloc[-1]
        sma_slope_v = sma25_slope20_pct.iloc[-1]
        bb_up_v = bb_up_1.iloc[-1]
        bb_dn_v = bb_dn_1.iloc[-1]

        plus_di_v = plus_di14.iloc[-1]
        minus_di_v = minus_di14.iloc[-1]
        di_diff_v = di_diff.iloc[-1]
        di_ratio_v = di_ratio.iloc[-1]

        # 底値上昇比率（BB_up_1 ÷ BB_dn_1）
        bb_range_ratio_v = bb_up_v / bb_dn_v if float(bb_dn_v) != 0.0 else np.nan

        if any(pd.isna(x) for x in [
            close_v, atr_v, adx_v, up_cnt, dn_cnt, sma_slope_v, bb_up_v, bb_dn_v,
            plus_di_v, minus_di_v, di_diff_v, di_ratio_v, bb_range_ratio_v
        ]):
            return {
                "Ticker": ticker,
                "Name": ticker_name_map.get(ticker, ""),
                "Close": float(close_v) if not pd.isna(close_v) else np.nan,
                "ATR20_pct": float(atr_v) if not pd.isna(atr_v) else np.nan,
                "ADX14": float(adx_v) if not pd.isna(adx_v) else np.nan,
                "BB_up_touch_cnt20": int(up_cnt) if not pd.isna(up_cnt) else np.nan,
                "BB_dn_touch_cnt20": int(dn_cnt) if not pd.isna(dn_cnt) else np.nan,
                "BB_up_1": float(bb_up_v) if not pd.isna(bb_up_v) else np.nan,
                "BB_dn_1": float(bb_dn_v) if not pd.isna(bb_dn_v) else np.nan,
                "SMA25_slope20_pct": float(sma_slope_v) if not pd.isna(sma_slope_v) else np.nan,
                "PLUS_DI14": float(plus_di_v) if not pd.isna(plus_di_v) else np.nan,
                "MINUS_DI14": float(minus_di_v) if not pd.isna(minus_di_v) else np.nan,
                "DI_diff": float(di_diff_v) if not pd.isna(di_diff_v) else np.nan,
                "DI_Ratio": float(di_ratio_v) if not pd.isna(di_ratio_v) else np.nan,
                "Bottom_Rise_Ratio": float(bb_range_ratio_v) if not pd.isna(bb_range_ratio_v) else np.nan,
                "Pass_ADX": False,
                "Pass_ATR": False,
                "Pass_BB": False,
                "Pass_SMA_Slope": False,
                "Pass_DI_Diff": False,
                "Pass_DI_Ratio": False,
                "Pass": False,
                "Reason": "nan_metrics",
            }

        # --- 条件判定（抽出条件と同一） ---
        pass_adx = float(adx_v) <= ADX_MAX
        pass_atr = (ATR_MIN_PCT <= float(atr_v) <= ATR_MAX_PCT)
        pass_bb = (int(up_cnt) >= BB_TOUCH_MIN) and (int(dn_cnt) >= BB_TOUCH_MIN)
        pass_sma = abs(float(sma_slope_v)) <= SMA_SLOPE_MAX_PCT
        pass_di_diff = float(di_diff_v) <= DI_DIFF_MAX
        pass_di_ratio = float(di_ratio_v) >= DI_RATIO_MIN

        passed = all([pass_adx, pass_atr, pass_bb, pass_sma, pass_di_diff, pass_di_ratio])

        return {
            "Ticker": ticker,
            "Name": ticker_name_map.get(ticker, ""),
            "Close": float(close_v),
            "ATR20_pct": float(atr_v),
            "ADX14": float(adx_v),
            "BB_up_touch_cnt20": int(up_cnt),
            "BB_dn_touch_cnt20": int(dn_cnt),
            "BB_up_1": float(bb_up_v),
            "BB_dn_1": float(bb_dn_v),
            "SMA25_slope20_pct": float(sma_slope_v),
            "PLUS_DI14": float(plus_di_v),
            "MINUS_DI14": float(minus_di_v),
            "DI_diff": float(di_diff_v),
            "DI_Ratio": float(di_ratio_v),                      # +DI14 / -DI14
            "Bottom_Rise_Ratio": float(bb_range_ratio_v),        # BB_up_1 / BB_dn_1
            "Pass_ADX": bool(pass_adx),
            "Pass_ATR": bool(pass_atr),
            "Pass_BB": bool(pass_bb),
            "Pass_SMA_Slope": bool(pass_sma),
            "Pass_DI_Diff": bool(pass_di_diff),
            "Pass_DI_Ratio": bool(pass_di_ratio),
            "Pass": bool(passed),
            "Reason": "",
        }

    except Exception:
        return {
            "Ticker": ticker,
            "Name": ticker_name_map.get(ticker, ""),
            "Close": np.nan,
            "ATR20_pct": np.nan,
            "ADX14": np.nan,
            "BB_up_touch_cnt20": np.nan,
            "BB_dn_touch_cnt20": np.nan,
            "BB_up_1": np.nan,
            "BB_dn_1": np.nan,
            "SMA25_slope20_pct": np.nan,
            "PLUS_DI14": np.nan,
            "MINUS_DI14": np.nan,
            "DI_diff": np.nan,
            "DI_Ratio": np.nan,
            "Bottom_Rise_Ratio": np.nan,
            "Pass_ADX": False,
            "Pass_ATR": False,
            "Pass_BB": False,
            "Pass_SMA_Slope": False,
            "Pass_DI_Diff": False,
            "Pass_DI_Ratio": False,
            "Pass": False,
            "Reason": "exception",
        }


def screen_candidates(raw_df: pd.DataFrame, tickers):
    rows = []
    for t in tickers:
        m = calc_latest_metrics_from_raw(raw_df, t)
        if m is not None:
            rows.append(m)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["ADX14", "ATR20_pct"], ascending=[True, False]).reset_index(drop=True)
    return df


def compute_all_metrics(raw_df: pd.DataFrame, tickers):
    """
    全銘柄分：計算できた指標を全部出す（抽出条件で落とさない）。
    Pass/Failと、各条件の判定列も含める（閾値調整用）。
    """
    rows = []
    for t in tickers:
        m = calc_latest_metrics_all_from_raw(raw_df, t)
        if m is None:
            continue
        rows.append(m)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["Pass", "Ticker"], ascending=[False, True]).reset_index(drop=True)
    return df


def save_chart_image_with_bb1sigma(raw_df: pd.DataFrame, ticker: str, out_dir: str = CHART_OUT_DIR):
    """
    既存運用を壊さないように：mplfinanceが使えるときだけ画像作成。
    チャートに BB(20,±1σ) を重ねる。
    """
    if not MPF_AVAILABLE:
        return None

    try:
        ohlcv = raw_df.loc[:, [(c, ticker) for c in ["Open", "High", "Low", "Close", "Volume"]]].copy()
        ohlcv.columns = ["Open", "High", "Low", "Close", "Volume"]
        ohlcv = ohlcv.dropna()
        if ohlcv.empty:
            return None
        ohlcv = ohlcv.tail(CHART_LOOKBACK_DAYS)

        close = ohlcv["Close"]
        bb_mid = close.rolling(20, min_periods=20).mean()
        bb_std = close.rolling(20, min_periods=20).std()
        bb_up_1 = bb_mid + bb_std
        bb_dn_1 = bb_mid - bb_std

        add_plots = [
            mpf.make_addplot(bb_up_1, panel=0),
            mpf.make_addplot(bb_dn_1, panel=0),
        ]

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{ticker}.png")

        mpf.plot(
            ohlcv,
            type="candle",
            mav=(5, 25, 75),
            volume=True,
            style="yahoo",
            title=ticker,
            addplot=add_plots,
            savefig=dict(fname=out_path, dpi=150, bbox_inches="tight"),
        )
        return out_path
    except Exception:
        return None


def notify(df: pd.DataFrame, raw_df: pd.DataFrame):
    title = (
        f"【ATRレンジ候補（ADX≤25 × ATR(1.8-4)% × BB±1σ>=3 × SMA傾き小 × "
        f"DI_diff≤{DI_DIFF_MAX:g} × DI比率≥{DI_RATIO_MIN:g}）】"
    )

    if df is None or df.empty:
        discord_send_text(f"{title} {now_jst():%m/%d %H:%M}\n対象なし")
        return

    lines = [f"{title} {now_jst():%m/%d %H:%M}"]
    lines.append("Ticker   名称       現在価格   ATR%   ADX   DIΔ   底値上昇比率   BB(+/-)   BB_dn1σ   BB_up1σ   SMA_slope%")

    for _, r in df.iterrows():
        t = r["Ticker"]
        name = r.get("Name", "")
        close = fp(r["Close"], 0)
        atr = fp(r["ATR20_pct"], 2)
        adx = fp(r["ADX14"], 1)
        did = fp(r["DI_diff"], 1)
        br = fp(r["Bottom_Rise_Ratio"], 3)  # BB比率なので小数は残す方が便利
        upc = str(int(r["BB_up_touch_cnt20"]))
        dnc = str(int(r["BB_dn_touch_cnt20"]))
        bb_dn = fp(r["BB_dn_1"], 0)
        bb_up = fp(r["BB_up_1"], 0)
        smas = fp(r["SMA25_slope20_pct"], 2)
        lines.append(
            f"{t:<7} {name:<8} {close:>8} {atr:>6} {adx:>6} {did:>5} {br:>12}   "
            f"{upc:>2}/{dnc:<2}   {bb_dn:>7}   {bb_up:>7}   {smas:>9}"
        )

    for part in chunk_text("\n".join(lines)):
        discord_send_text(part)

    # 画像（BB±1σ線入り）
    if not MPF_AVAILABLE:
        return
    for _, r in df.head(CHART_TOP_N).iterrows():
        t = r["Ticker"]
        name = r.get("Name", "")

        # 現在価格 → BB(±1σ) → 底値上昇比率（BB比率） → 以降は今の順番
        desc = (
            f"現在価格:{fp(r['Close'],0)}  "
            f"BB(±1σ):{fp(r['BB_dn_1'],0)}–{fp(r['BB_up_1'],0)}  "
            f"底値上昇比率:{fp(r['Bottom_Rise_Ratio'],3)}  "
            f"ATR%:{fp(r['ATR20_pct'],2)}  "
            f"ADX:{fp(r['ADX14'],1)}  "
            f"DI_diff:{fp(r['DI_diff'],1)}  "
            f"BB(+/-):{int(r['BB_up_touch_cnt20'])}/{int(r['BB_dn_touch_cnt20'])}  "
            f"SMA_slope%:{fp(r['SMA25_slope20_pct'],2)}"
        )

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

    # yfinanceが存在しない銘柄で落ちないように（運用安定）
    if raw is None or raw.empty:
        discord_send_text(f"【ATRレンジ候補】 {now_jst():%m/%d %H:%M}\nデータ取得失敗")
        return

    # ===== 全銘柄 指標CSV出力（閾値調整用）=====
    all_df = compute_all_metrics(raw, tickers)
    if all_df is not None and not all_df.empty:
        os.makedirs(METRICS_OUT_DIR, exist_ok=True)
        ts = now_jst().strftime("%Y%m%d_%H%M")
        out_path = os.path.join(METRICS_OUT_DIR, f"{METRICS_PREFIX}_{ts}.csv")
        all_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] metrics csv saved: {out_path}")
    else:
        print("[WARN] metrics csv not created (no rows).", file=sys.stderr)

    # ===== 抽出 → 通知 =====
    df = screen_candidates(raw, tickers)
    notify(df, raw)


if __name__ == "__main__":
    main()
