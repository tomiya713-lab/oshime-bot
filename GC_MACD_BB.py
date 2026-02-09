import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# チャート用
try:
    import mplfinance as mpf  # type: ignore
    MPF_AVAILABLE = True
except Exception:
    MPF_AVAILABLE = False


# ===== タイムゾーン／環境変数 =====
TZ_OFFSET = 9  # JST

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "180"))

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
DISCORD_ENABLED = bool(DISCORD_WEBHOOK_URL)

FORCE_RUN = os.getenv("FORCE_RUN", "0") == "1"

# ===== 抽出ロジックパラメータ =====
# 1) 短期（SMA5×SMA25）…「直前狙い（PreGC）」と「クロス当日（GC）」を両方拾う
SMA_SHORT = 5
SMA_LONG = 25

# “直前狙い”用：SMA5/SMA25 がこの比率以上（例: 0.90）
SMA_RATIO_MIN = float(os.getenv("SMA_RATIO_MIN", "0.90"))

# 「SMAが上向き」の定義：SMA25が上向き（おすすめ）
REQUIRE_SMA25_UP = os.getenv("REQUIRE_SMA25_UP", "1") == "1"

# 2) 王道GC（SMA25×SMA75）…クロス当日を拾う
SMA_TREND = 75  # 固定

# MACD (12,26,9) 固定
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# 週足BB：20週 + 2σ（“条件”には使わないが、metricsには残す）
WBB_PERIOD_WEEKS = 20
WBB_SIGMA = 2.0

# チャート設定
CHART_OUT_DIR = "charts"
CHART_LOOKBACK_DAYS = 90   # 直近◯営業日分を描画
CHART_TOP_N = 8            # 画像を出す最大銘柄数

# 指標CSV出力（全銘柄・閾値調整用）
METRICS_OUT_DIR = os.getenv("METRICS_OUT_DIR", "reports")
METRICS_PREFIX = os.getenv("METRICS_PREFIX", "gc_macd_bb_metrics")


def now_jst() -> datetime:
    return datetime.utcnow() + timedelta(hours=TZ_OFFSET)


def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5


def chunk_text(text: str, limit: int = 1900) -> List[str]:
    """Discord 2000文字制限対策。行単位で分割。"""
    out: List[str] = []
    buf: List[str] = []
    size = 0
    for line in text.splitlines():
        add = len(line) + 1
        if size + add > limit:
            out.append("\n".join(buf))
            buf = [line]
            size = add
        else:
            buf.append(line)
            size += add
    if buf:
        out.append("\n".join(buf))
    return out


def discord_send_content(msg: str) -> None:
    """Webhook未設定でも落ちないようにしておく。"""
    if not DISCORD_ENABLED:
        print("[INFO] DISCORD_WEBHOOK_URL not set. Message below was NOT sent to Discord:", file=sys.stderr)
        print(msg)
        return

    try:
        r = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": msg},
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        if r.status_code >= 300:
            print(f"[WARN] Discord post failed: {r.status_code} {r.text}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Discord post exception: {e}", file=sys.stderr)


def discord_send_image_file(file_path: str, title: str, description: Optional[str] = None) -> None:
    """画像ファイルを添付して送信（外部URL不要）。"""
    if not DISCORD_ENABLED:
        print(f"[INFO] (image not sent) {title}: {file_path}", file=sys.stderr)
        return

    filename = os.path.basename(file_path)
    embed: Dict[str, Any] = {
        "title": title,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if description:
        embed["description"] = description
    embed["image"] = {"url": f"attachment://{filename}"}

    def json_dumps(obj) -> str:
        import json
        return json.dumps(obj, ensure_ascii=False)

    try:
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "image/png")}
            data = {"payload_json": json_dumps({"embeds": [embed]})}
            r = requests.post(DISCORD_WEBHOOK_URL, files=files, data=data, timeout=30)
            if r.status_code >= 300:
                print(f"[WARN] Discord image upload failed: {r.status_code} {r.text}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Discord image send exception: {e}", file=sys.stderr)


def send_long_text(msg: str) -> None:
    for part in chunk_text(msg):
        discord_send_content(part)


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


def load_tickers() -> List[str]:
    return list(dict.fromkeys([t.strip() for t in nikkei225_tickers if str(t).strip()]))


def fetch_market_data(tickers: List[str], lookback_days: int = LOOKBACK_DAYS) -> Optional[pd.DataFrame]:
    """yfinanceでOHLCVを一括取得（Adj Close不使用）"""
    try:
        period = f"{int(lookback_days)}d"
        df = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=False,
            group_by="column",
            threads=True,
            progress=False,
        )
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        print(f"[ERROR] fetch_market_data failed: {e}", file=sys.stderr)
        return None


def _extract_series(raw_df: pd.DataFrame, ticker: str, col: str) -> Optional[pd.Series]:
    try:
        if isinstance(raw_df.columns, pd.MultiIndex):
            s = raw_df[(col, ticker)].dropna().copy()
        else:
            s = raw_df[col].dropna().copy()
        if s is None or len(s) == 0:
            return None
        return s
    except Exception:
        return None


def latest_gc_signals_from_raw(raw_df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """
    直近日のシグナルを計算して返す。

    ✅要望反映
    - “直前狙い”も残す（PreGC 5x25）
    - クロス当日も拾う（GC 5x25 / GC 25x75）
    - MACDは「MACD > Signal」のみ（0以上条件は撤廃）
    - 週足BBブレイク条件は削除（ただしmetrics用に週足BB距離は計算）
    - metrics csvに以下を追加：
        SMA5_prev, SMA25_prev, SMA75_prev,
        MACD_hist,
        WBB_dist_pct,
        Volume, VolAvg5
    """
    out: Dict[str, Any] = {"Ticker": ticker}

    close = _extract_series(raw_df, ticker, "Close")
    vol = _extract_series(raw_df, ticker, "Volume")

    if close is None or len(close) < max(SMA_TREND, SMA_LONG) + 3:
        out.update({
            "ok": False,
            "gc_pre_5_25": False,
            "gc_cross_5_25": False,
            "gc_cross_25_75": False,
            "macd_bullish": False,
            "Signals": "",
            "Signal_Count": 0,
            "Reason": "insufficient_close",
        })
        return out

    # --- SMA ---
    sma5 = close.rolling(window=SMA_SHORT, min_periods=SMA_SHORT).mean()
    sma25 = close.rolling(window=SMA_LONG, min_periods=SMA_LONG).mean()
    sma75 = close.rolling(window=SMA_TREND, min_periods=SMA_TREND).mean()

    sma5_t, sma25_t, sma75_t = float(sma5.iloc[-1]), float(sma25.iloc[-1]), float(sma75.iloc[-1])
    sma5_p, sma25_p, sma75_p = float(sma5.iloc[-2]), float(sma25.iloc[-2]), float(sma75.iloc[-2])

    out["SMA5"] = sma5_t
    out["SMA25"] = sma25_t
    out["SMA75"] = sma75_t
    out["SMA5_prev"] = sma5_p
    out["SMA25_prev"] = sma25_p
    out["SMA75_prev"] = sma75_p

    ratio_t = (sma5_t / sma25_t) if sma25_t != 0.0 else np.nan
    ratio_p = (sma5_p / sma25_p) if sma25_p != 0.0 else np.nan
    out["SMA_ratio"] = float(ratio_t) if not np.isnan(ratio_t) else np.nan
    out["SMA_ratio_prev"] = float(ratio_p) if not np.isnan(ratio_p) else np.nan

    sma25_up = bool(sma25_t > sma25_p)
    out["SMA25_up"] = bool(sma25_up)

    # --- “直前狙い”（SMA5×SMA25） ---
    gc_pre_5_25 = (
        (sma5_t < sma25_t) and
        (not np.isnan(ratio_t) and ratio_t >= SMA_RATIO_MIN) and
        (not np.isnan(ratio_p) and ratio_t > ratio_p) and
        (sma25_up if REQUIRE_SMA25_UP else True)
    )

    # --- “クロス当日”（SMA5×SMA25） ---
    gc_cross_5_25 = (sma5_p <= sma25_p) and (sma5_t > sma25_t)

    # --- “クロス当日”（SMA25×SMA75：王道GC） ---
    gc_cross_25_75 = (sma25_p <= sma75_p) and (sma25_t > sma75_t)

    out["gc_pre_5_25"] = bool(gc_pre_5_25)
    out["gc_cross_5_25"] = bool(gc_cross_5_25)
    out["gc_cross_25_75"] = bool(gc_cross_25_75)

    # --- MACD (12,26,9) ---
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    out["MACD"] = float(macd_line.iloc[-1])
    out["MACD_signal"] = float(macd_signal.iloc[-1])
    out["MACD_hist"] = float(macd_hist.iloc[-1])  # 追加

    macd_bullish = float(macd_line.iloc[-1]) > float(macd_signal.iloc[-1])  # 0以上条件なし
    out["macd_bullish"] = bool(macd_bullish)

    # --- 出来高 / 5日平均 ---
    if vol is not None and len(vol) >= 6:
        out["Volume"] = float(vol.iloc[-1])
        out["VolAvg5"] = float(vol.tail(5).mean())
    else:
        out["Volume"] = np.nan
        out["VolAvg5"] = np.nan

    # --- 週足BB（条件には使わない。metrics用） ---
    try:
        weekly_close = close.to_frame("Close").resample("W-FRI").last()["Close"].dropna()
        if len(weekly_close) >= WBB_PERIOD_WEEKS + 2:
            w_ma20 = weekly_close.rolling(window=WBB_PERIOD_WEEKS, min_periods=WBB_PERIOD_WEEKS).mean()
            w_std20 = weekly_close.rolling(window=WBB_PERIOD_WEEKS, min_periods=WBB_PERIOD_WEEKS).std()
            w_upper = w_ma20 + WBB_SIGMA * w_std20

            wclose = float(weekly_close.iloc[-1])
            wbb_upper = float(w_upper.iloc[-1])
            out["WClose"] = wclose
            out["WBB_upper"] = wbb_upper
            out["WBB_dist_pct"] = (wclose / wbb_upper - 1.0) if (wbb_upper and not np.isnan(wbb_upper)) else np.nan  # 追加
        else:
            out["WClose"] = np.nan
            out["WBB_upper"] = np.nan
            out["WBB_dist_pct"] = np.nan
    except Exception:
        out["WClose"] = np.nan
        out["WBB_upper"] = np.nan
        out["WBB_dist_pct"] = np.nan

    # --- シグナルまとめ（Discord表示用） ---
    signals: List[str] = []
    if gc_pre_5_25:
        signals.append("PreGC 5x25")
    if gc_cross_5_25:
        signals.append("GC 5x25")
    if gc_cross_25_75:
        signals.append("GC 25x75")

    out["Signals"] = " / ".join(signals)
    out["Signal_Count"] = int(len(signals))

    # 最終OK：GC系（どれか） AND MACD強気
    ok = bool(macd_bullish and len(signals) > 0)
    out["ok"] = ok
    out["Reason"] = "" if ok else "conditions_not_met"
    return out


# ===== チャート生成 =====
def save_chart_image_from_raw(raw_df: pd.DataFrame, ticker: str, out_dir: str = CHART_OUT_DIR) -> Optional[str]:
    if not MPF_AVAILABLE:
        return None

    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{ticker}.png")

        if isinstance(raw_df.columns, pd.MultiIndex):
            use = raw_df.loc[:, [(c, ticker) for c in ["Open", "High", "Low", "Close", "Volume"]]].copy()
            use.columns = ["Open", "High", "Low", "Close", "Volume"]
        else:
            use = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()

        use = use.dropna().tail(CHART_LOOKBACK_DAYS)

        mpf.plot(
            use,
            type="candle",
            mav=(5, 25, 75),
            volume=True,
            style="yahoo",
            savefig=dict(fname=out_path, dpi=140, bbox_inches="tight"),
        )
        return out_path
    except Exception as e:
        print(f"[WARN] mplfinance plot failed for {ticker}: {e}", file=sys.stderr)
        return None


def fp(x: Any, nd: int = 2) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


# ===== 全銘柄CSV（閾値調整用） =====
def compute_all_metrics(raw_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in tickers:
        res = latest_gc_signals_from_raw(raw_df, t)
        res["Name"] = ticker_name_map.get(t, "")
        if not res.get("Signals"):
            res["Signals"] = ""
        rows.append(res)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df.empty:
        return df

    # 並び：ok優先 → Signal数 → MACD_hist → SMA_ratio
    df = df.sort_values(
        ["ok", "Signal_Count", "MACD_hist", "SMA_ratio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return df


# ===== シグナル抽出 =====
def screen_gc_signals(raw_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for t in tickers:
        res = latest_gc_signals_from_raw(raw_df, t)
        if not res.get("ok", False):
            continue
        rows.append(res)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if df.empty:
        return df

    df = df.sort_values(
        ["Signal_Count", "MACD_hist", "SMA_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return df


def notify(df: pd.DataFrame, raw_df: pd.DataFrame) -> None:
    title = (
        f"【GCシグナル + MACD強気】 {now_jst():%m/%d %H:%M}\n"
        f"- PreGC 5x25: SMA5<SMA25, SMA25上向き={1 if REQUIRE_SMA25_UP else 0}, "
        f"SMA5/SMA25≥{SMA_RATIO_MIN:.2f}, ratio改善\n"
        f"- GC 5x25: 前日までSMA5<=SMA25 → 当日SMA5>SMA25\n"
        f"- GC 25x75: 前日までSMA25<=SMA75 → 当日SMA25>SMA75\n"
        f"- MACD: MACD > Signal（0以上条件なし）"
    )

    if df is None or df.empty:
        discord_send_content(f"{title}\n\n対象なし")
        return

    lines = [title, f"\n候補：{len(df)}件", ""]
    for _, r in df.iterrows():
        t = r.get("Ticker", "")
        name = ticker_name_map.get(str(t), "")
        signals = r.get("Signals", "")
        lines.append(
            f"{t} {name} | {signals} | "
            f"ratio={fp(r.get('SMA_ratio'),3)} (prev {fp(r.get('SMA_ratio_prev'),3)}) "
            f"SMA5={fp(r.get('SMA5'),2)} SMA25={fp(r.get('SMA25'),2)} SMA75={fp(r.get('SMA75'),2)} "
            f"MACD={fp(r.get('MACD'),3)} Sig={fp(r.get('MACD_signal'),3)} Hist={fp(r.get('MACD_hist'),3)} "
            f"Vol={fp(r.get('Volume'),0)} VolAvg5={fp(r.get('VolAvg5'),0)} "
            f"WBBdist={fp(r.get('WBB_dist_pct'),3)}"
        )

    send_long_text("\n".join(lines))

    # --- チャート画像部 ---
    if not MPF_AVAILABLE:
        print("[INFO] mplfinance not installed; charts will not be generated.", file=sys.stderr)
        return

    top = df.head(CHART_TOP_N)
    for _, r in top.iterrows():
        t = str(r.get("Ticker", ""))
        name = ticker_name_map.get(t, "")
        ttl = f"{t} {name}".strip()
        desc = (
            f"{r.get('Signals','')} | "
            f"ratio:{fp(r.get('SMA_ratio'),3)} (prev {fp(r.get('SMA_ratio_prev'),3)})  "
            f"SMA5:{fp(r.get('SMA5'),2)} SMA25:{fp(r.get('SMA25'),2)} SMA75:{fp(r.get('SMA75'),2)}  "
            f"MACD:{fp(r.get('MACD'),3)} Sig:{fp(r.get('MACD_signal'),3)} Hist:{fp(r.get('MACD_hist'),3)}  "
            f"Vol:{fp(r.get('Volume'),0)} VolAvg5:{fp(r.get('VolAvg5'),0)}  "
            f"WBBdist:{fp(r.get('WBB_dist_pct'),3)}"
        )

        img_path = save_chart_image_from_raw(raw_df, t, out_dir=CHART_OUT_DIR)
        if img_path:
            discord_send_image_file(img_path, title=ttl, description=desc)


def main() -> None:
    now = now_jst()
    if not FORCE_RUN and is_weekend(now):
        print(f"[SKIP] {now:%F %R} 週末のためスキップ（FORCE_RUN=1で強制実行）")
        return

    tickers = load_tickers()
    if not tickers:
        print("[ERROR] nikkei225_tickers is empty.", file=sys.stderr)
        return

    raw = fetch_market_data(tickers, lookback_days=LOOKBACK_DAYS)
    if raw is None or raw.empty:
        discord_send_content(f"【GCシグナル + MACD強気】 {now_jst():%m/%d %H:%M}\nデータ取得失敗")
        return

    # ===== 全銘柄 指標CSV出力（閾値調整用） =====
    metrics_df = compute_all_metrics(raw, tickers)
    if metrics_df is not None and not metrics_df.empty:
        os.makedirs(METRICS_OUT_DIR, exist_ok=True)
        ts = now_jst().strftime("%Y%m%d_%H%M")
        out_path = os.path.join(METRICS_OUT_DIR, f"{METRICS_PREFIX}_{ts}.csv")
        metrics_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] metrics csv saved: {out_path}")
    else:
        print("[WARN] metrics csv not created (no rows).", file=sys.stderr)

    # ===== 抽出 → 通知 =====
    df = screen_gc_signals(raw, tickers)
    notify(df, raw)


if __name__ == "__main__":
    main()
