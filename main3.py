import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
import requests

# チャート用
try:
    import mplfinance as mpf
    MPF_AVAILABLE = True
except ImportError:
    MPF_AVAILABLE = False

# ===== タイムゾーン／環境変数 =====
TZ_OFFSET = 9  # JST

LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "180"))
TICKERS_CSV = os.getenv("TICKERS_CSV", "").strip()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
DISCORD_ENABLED = bool(DISCORD_WEBHOOK_URL)

FORCE_RUN = os.getenv("FORCE_RUN", "0") == "1"

# ===== 押し目抽出パラメータ（厳格版） =====
DROP_MAX = 15.0                 # ピークから最新値までの許容下落率 ≤ 15%
EXPECTED_RISE_MIN = 3.0         # Return_%（ピークまでの上昇余地）≥ 3%
PULLBACK_BAND_PCT = 2.0         # 最新終値が押し目安値の +2%以内なら押し目帯
WITHIN_UPPER = 1.0 + PULLBACK_BAND_PCT / 100.0

# 以前の「Return_% ≥ 5% なら通過」は、押し目から大きく反発済みの銘柄も拾うためデフォルトOFF。
# どうしても戻り余地重視の候補も拾いたい場合だけ、環境変数 USE_RETURN_OR=1 を指定する。
USE_RETURN_OR = os.getenv("USE_RETURN_OR", "0") == "1"
EXP_OR = 5.0                    # USE_RETURN_OR=1 のときだけ使う

WEEKLY_MA2_FILTER = True        # 週足MA(2)が下向きなら除外
WEEKLY_RESAMPLE = "W-FRI"       # 週足は金曜終値でリサンプル
SCREEN_WINDOWS = (60, 30)       # 60日・30日で抽出し、同一銘柄はReturn_%が大きい方を採用

PASS_REASON_BAND = "押し目帯"
PASS_REASON_EXP = "Return_≥5%"

# チャート説明用（選定条件には使わない）
RSI_PERIOD = 14
# チャート設定
CHART_OUT_DIR = "charts"
CHART_LOOKBACK_DAYS = 90   # 直近◯営業日分を描画
CHART_TOP_N = 8            # 画像を出す最大銘柄数


# ===== ユーティリティ =====
def now_jst() -> datetime:
    return datetime.utcnow() + timedelta(hours=TZ_OFFSET)


def is_weekend(dt: datetime) -> bool:
    return dt.weekday() >= 5  # 5=Sat, 6=Sun


def chunk_text(text: str, limit: int = 1900):
    """Discord 2000文字制限対策。行単位で分割。"""
    out, buf, size = [], [], 0
    for line in text.splitlines():
        add = len(line) + 1
        if size + add > limit:
            out.append("\n".join(buf))
            buf, size = [line], add
        else:
            buf.append(line)
            size += add
    if buf:
        out.append("\n".join(buf))
    return out


# ===== Discord 送信 =====
def discord_send_content(msg: str):
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


def discord_send_image_file(file_path: str, title: str, description: str | None = None):
    """画像ファイルを添付して送信（外部URL不要）。"""
    if not DISCORD_ENABLED:
        print(f"[INFO] (image not sent) {title}: {file_path}", file=sys.stderr)
        return

    filename = os.path.basename(file_path)
    embed = {
        "title": title,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if description:
        embed["description"] = description
    # 添付ファイルは attachment://<filename> で参照する
    embed["image"] = {"url": f"attachment://{filename}"}

    try:
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "image/png")}
            data = {"payload_json": json_dumps({"embeds": [embed]})}
            r = requests.post(DISCORD_WEBHOOK_URL, files=files, data=data, timeout=30)
            if r.status_code >= 300:
                print(f"[WARN] Discord image upload failed: {r.status_code} {r.text}", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Discord image send exception: {e}", file=sys.stderr)


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


def send_long_text(msg: str):
    for part in chunk_text(msg):
        discord_send_content(part)


# ===== ティッカー一覧 =====

# フォールバックとして日経225ティッカー
NIKKEI225_TICKERS = [
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
    '9107.T','9201.T','9202.T','9301.T','9501.T','9502.T','9503.T','9531.T','9532.T',
]

# 銘柄名（通知用）

TICKER_NAME_MAP = {
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

def load_tickers():
    if TICKERS_CSV and os.path.exists(TICKERS_CSV):
        df = pd.read_csv(TICKERS_CSV)
        col = None
        for c in df.columns:
            if c.lower() in ("ticker", "symbol", "code"):
                col = c
                break
        if col:
            tickers = [str(x).strip() for x in df[col].dropna().unique().tolist()]
            if tickers:
                return tickers
    return NIKKEI225_TICKERS


# ===== yfinance データ取得 & 押し目抽出用データ整形 =====
def fetch_market_data(tickers, lookback_days=LOOKBACK_DAYS):
    end_dt = (now_jst().date() + timedelta(days=1)).isoformat()
    start_dt = (now_jst().date() - timedelta(days=lookback_days)).isoformat()

    raw = yf.download(
        tickers,
        start=start_dt,
        end=end_dt,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",  # (field, ticker) の MultiIndex
        threads=True,
    )
    return raw


def latest_rsi_from_raw(raw_df: pd.DataFrame, ticker: str, period: int = RSI_PERIOD):
    """raw_df から指定ティッカーの終値で RSI を算出し、直近値を返す。選定条件には使わない。"""
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


def extract_price_frames(raw_df: pd.DataFrame, tickers):
    """yfinanceのrawデータから Close / High / Low をDataFrameで取り出す。"""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if isinstance(raw_df.columns, pd.MultiIndex):
        try:
            close = raw_df["Close"].copy()
            high = raw_df["High"].copy()
            low = raw_df["Low"].copy()
        except Exception as e:
            raise RuntimeError(f"yfinance returned missing price columns: {e}")
    else:
        # 単一ティッカーでMultiIndexにならなかった場合
        ticker = tickers[0] if tickers else "Ticker"
        close = raw_df[["Close"]].rename(columns={"Close": ticker}).copy()
        high = raw_df[["High"]].rename(columns={"High": ticker}).copy()
        low = raw_df[["Low"]].rename(columns={"Low": ticker}).copy()

    # TICKERS_CSVなどで指定した銘柄だけに揃える。存在しない列は自然に除外。
    keep = [t for t in tickers if t in close.columns and t in high.columns and t in low.columns]
    return close[keep], high[keep], low[keep]


def _weekly_ma2_is_down(close_s: pd.Series) -> bool:
    """
    週足（W-FRI）終値→2週SMAを作成し、直近MA2 < 1週前MA2 なら“下向き”。
    データ不足やNaNがある場合は False（除外しない）として扱う。
    """
    s = close_s.dropna().resample(WEEKLY_RESAMPLE).last()
    if len(s) < 3:
        return False
    ma2 = s.rolling(2, min_periods=2).mean()
    last = ma2.iloc[-1]
    prev = ma2.iloc[-2]
    if np.isnan(last) or np.isnan(prev):
        return False
    return bool(last < prev)


def compute_one_ticker(close_s: pd.Series, high_s: pd.Series, low_s: pd.Series, window_days=60):
    """
    厳格版の押し目抽出ロジック。
      ベース: Drop≤15%, Return_%≥3%
      除外   : latest == pull_low
      採用   : pull_low < latest ≤ pull_low*(1 + PULLBACK_BAND_PCT%)
      任意   : USE_RETURN_OR=1 の場合のみ Return_% ≥ EXP_OR でも通過
      週足   : 2週SMAが下向きなら除外

    注意:
      USE_RETURN_OR=0 がデフォルト。
      これにより、押し目安値から大きく反発済みの“戻り売りっぽい銘柄”を除外する。
    """
    ticker = getattr(close_s, "name", "")
    try:
        # Close / High / Low を同じ日付で揃える
        price = pd.concat(
            [close_s.rename("Close"), high_s.rename("High"), low_s.rename("Low")],
            axis=1,
        ).dropna()

        if len(price) < window_days + 2:
            return None

        close_all = price["Close"]
        high_all = price["High"]
        low_all = price["Low"]

        # 対象期間
        look = price.iloc[-window_days:]
        if look.empty:
            return None

        # ピーク（期間内の最高値）
        peak_idx = look["High"].idxmax()
        peak_val = float(look.loc[peak_idx, "High"])

        # ピーク後の最安値（押し目）
        after_peak = look.loc[look.index > peak_idx, "Low"]
        if after_peak.empty:
            return None
        pull_idx = after_peak.idxmin()
        pull_val = float(after_peak.loc[pull_idx])

        # 最新
        latest_idx = close_all.index[-1]
        latest_val = float(close_all.iloc[-1])
        prev_val = float(close_all.iloc[-2]) if len(close_all) >= 2 else np.nan

        # 指標
        drop_pct = (1.0 - latest_val / peak_val) * 100.0
        expected_rise_pct = (peak_val / latest_val - 1.0) * 100.0
        delta_from_pull_pct = (latest_val / pull_val - 1.0) * 100.0

        # グローバル除外：最新終値が押し目安値と完全一致する場合は除外
        if np.isclose(latest_val, pull_val, rtol=0.0, atol=1e-6):
            return None

        # 押し目帯条件
        within_band = (latest_val > pull_val) and (latest_val <= pull_val * WITHIN_UPPER)

        pass_reason = None
        if within_band:
            pass_reason = PASS_REASON_BAND
        elif USE_RETURN_OR and expected_rise_pct >= EXP_OR:
            # デフォルトではOFF。戻り余地重視の候補も拾いたい場合だけ使う。
            pass_reason = PASS_REASON_EXP

        # ベース条件 + 押し目帯成立
        if not (
            drop_pct <= DROP_MAX
            and expected_rise_pct >= EXPECTED_RISE_MIN
            and pass_reason is not None
        ):
            return None

        # 週足MA(2)フィルター
        if WEEKLY_MA2_FILTER and _weekly_ma2_is_down(close_all):
            return None

        return {
            "Ticker": ticker,
            "Peak_Date": peak_idx.date(),
            "Peak_High": round(peak_val, 2),
            "Pullback_Date": pull_idx.date(),
            "Pullback_Low": round(pull_val, 2),
            "Latest_Date": latest_idx.date(),
            "Latest_Close": round(latest_val, 2),
            "Prev_Close": round(prev_val, 2) if not np.isnan(prev_val) else np.nan,
            "Return_%": round(expected_rise_pct, 2),
            "Drop_From_Peak_%": round(drop_pct, 2),
            "Delta_from_Pull_%": round(delta_from_pull_pct, 2),
            "Within_(pull, +2%]": within_band,
            "OR_Return_ge_5%": USE_RETURN_OR and expected_rise_pct >= EXP_OR,
            "Pass_Reason": pass_reason,
            "Window": window_days,
        }
    except Exception as e:
        print(f"[WARN] compute_one_ticker failed for {ticker}: {e}", file=sys.stderr)
        return None


def find_pullback_candidates(close_df: pd.DataFrame, high_df: pd.DataFrame, low_df: pd.DataFrame, window_days=30):
    rows = []
    for ticker in close_df.columns:
        res = compute_one_ticker(
            close_df[ticker],
            high_df[ticker],
            low_df[ticker],
            window_days=window_days,
        )
        if res:
            rows.append(res)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("Return_%", ascending=False).reset_index(drop=True)
    return df


def screen_pullback(raw_df: pd.DataFrame, tickers):
    """
    60日・30日で押し目候補を抽出し、同一ティッカーはReturn_%が大きい方を採用。
    """
    close, high, low = extract_price_frames(raw_df, tickers)
    if close.empty:
        return pd.DataFrame()

    results = []
    for w in SCREEN_WINDOWS:
        df = find_pullback_candidates(close, high, low, window_days=w)
        if not df.empty:
            df["Window"] = w
            results.append(df)

    if not results:
        return pd.DataFrame()

    cat = pd.concat(results, ignore_index=True)
    best = (
        cat.sort_values(["Ticker", "Return_%"], ascending=[True, False])
           .groupby("Ticker", as_index=False)
           .first()
           .sort_values("Return_%", ascending=False)
           .reset_index(drop=True)
    )
    return best


# ===== チャート生成 =====
def save_chart_image_from_raw(raw_df: pd.DataFrame, ticker: str, out_dir: str = CHART_OUT_DIR):
    """yfinanceのrawデータから対象ティッカーのローソク＋移動平均チャートをPNG保存。"""
    if not MPF_AVAILABLE:
        return None

    need_cols = ["Open", "High", "Low", "Close", "Volume"]
    try:
        if isinstance(raw_df.columns, pd.MultiIndex):
            use = raw_df.loc[:, [(c, ticker) for c in need_cols]].copy()
            use.columns = need_cols
        else:
            # 単一ティッカー想定
            use = raw_df[need_cols].copy()
    except Exception:
        return None

    use = use.dropna()
    if use.empty:
        return None

    # 直近数十日分に絞る
    use = use.tail(CHART_LOOKBACK_DAYS)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.png")

    try:
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


# ===== 押し目通知 =====
def notify(df: pd.DataFrame, raw_df: pd.DataFrame):
    # --- テキスト部 ---
    if df is None or df.empty:
        msg = (
            f"【押し目スクリーニング】{now_jst():%m/%d %H:%M}\n"
            f"条件: Drop≤{DROP_MAX:.0f}%・Return≥{EXPECTED_RISE_MIN:.0f}%・"
            f"押し目安値+{PULLBACK_BAND_PCT:.0f}%以内"
            f"{' or Return≥' + str(int(EXP_OR)) + '%' if USE_RETURN_OR else ''}・"
            f"週足MA(2)下向き除外\n"
            f"該当銘柄はありませんでした。"
        )
        send_long_text(msg)
        return

    header = (
        f"【押し目スクリーニング】{now_jst():%m/%d %H:%M}\n"
        f"条件: Drop≤{DROP_MAX:.0f}%・Return≥{EXPECTED_RISE_MIN:.0f}%・"
        f"押し目安値+{PULLBACK_BAND_PCT:.0f}%以内"
        f"{' or Return≥' + str(int(EXP_OR)) + '%' if USE_RETURN_OR else ''}・"
        f"週足MA(2)下向き除外\n"
        f"抽出: {len(df)} 銘柄\n"
        f"------------------------------"
    )

    lines = [header]

    def fp(v, digits=1):
        try:
            if pd.isna(v):
                return "-"
            return f"{float(v):.{digits}f}"
        except Exception:
            return "-"

    def fdate(v):
        try:
            if hasattr(v, "strftime"):
                return v.strftime("%m/%d")
            return pd.to_datetime(v).strftime("%m/%d")
        except Exception:
            return "-"

    for _, r in df.iterrows():
        t = r["Ticker"]
        name = TICKER_NAME_MAP.get(t, "")
        reason = str(r.get("Pass_Reason", "")) or "-"
        line = (
            f"{t:<8} {name:<8} [{reason}]  "
            f"Return: {fp(r.get('Return_%'), 1):>5}%  "
            f"Drop: {fp(r.get('Drop_From_Peak_%'), 1):>5}%  "
            f"ΔPull: {fp(r.get('Delta_from_Pull_%'), 1):>5}%  "
            f"Win: {int(r.get('Window') or 0):>2}d  "
            f"Pull: {fdate(r.get('Pullback_Date'))}"
        )
        lines.append(line)

    msg = "\n".join(lines)
    send_long_text(msg)

    # --- チャート画像部 ---
    if not MPF_AVAILABLE:
        print("[INFO] mplfinance not installed; charts will not be generated.", file=sys.stderr)
        return

    top = df.head(CHART_TOP_N)
    for _, r in top.iterrows():
        t = r["Ticker"]
        name = TICKER_NAME_MAP.get(t, "")
        reason = str(r.get("Pass_Reason", "")) or "-"
        title = f"{t} {name} [{reason}]".strip()

        rsi = latest_rsi_from_raw(raw_df, t, period=RSI_PERIOD)
        rsi_text = "-" if rsi is None or not np.isfinite(rsi) else f"{rsi:.0f}"

        desc = (
            f"Return: {fp(r.get('Return_%'), 1)}%  "
            f"Drop: {fp(r.get('Drop_From_Peak_%'), 1)}%  "
            f"ΔPull: {fp(r.get('Delta_from_Pull_%'), 1)}%  "
            f"Win: {int(r.get('Window') or 0)}d  "
            f"RSI14: {rsi_text}\n"
            f"Latest: {fp(r.get('Latest_Close'), 0)}  "
            f"PullLow: {fp(r.get('Pullback_Low'), 0)}  "
            f"Peak: {fp(r.get('Peak_High'), 0)}"
        )

        img_path = save_chart_image_from_raw(raw_df, t, out_dir=CHART_OUT_DIR)
        if img_path:
            discord_send_image_file(img_path, title=title, description=desc)


# ===== メイン =====
def main():
    now = now_jst()
    if not FORCE_RUN and is_weekend(now):
        print(f"[SKIP] {now:%F %R} 週末のためスキップ（FORCE_RUN=1で強制実行）")
        return

    tickers = load_tickers()
    raw = fetch_market_data(tickers, lookback_days=LOOKBACK_DAYS)
    df = screen_pullback(raw, tickers)
    notify(df, raw)


if __name__ == "__main__":
    main()
