
import os
import sys
import json
import math
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf

# ===== 設定（必要に応じて変更） =====
DEPTH_MIN_PCT      = float(os.getenv("DEPTH_MIN_PCT", "0.06"))  # Peak→Low の下落率 下限（例 6%）
ATR20_MULT_MIN     = float(os.getenv("ATR20_MULT_MIN", "1.0"))  # Peak→Low の下落幅が ATR20 の何倍以上か
SMA_TOUCH_TOL      = float(os.getenv("SMA_TOUCH_TOL", "0.02"))  # SMA25 へのタッチ許容 ±2%
DAYS_SINCE_LOW_MIN = int(os.getenv("DAYS_SINCE_LOW_MIN", "3"))  # Low からの経過最小日数
TZ_OFFSET = 9  # JST
REBOUND_MIN = 0.01       # 反発率 >= 1%
REBOUND_MAX = 0.04       # 反発率 <= 4%
DROP_MAX = 15.0         # ピークからの許容下落率 <= 15%
DAYS_SINCE_MIN = 2      # 押し目から最新までの営業日数 >= 2
EXPECTED_RISE_MIN = 3.0 # 期待上昇率 >= 3%
SMA_WINDOW = 25
TOP_N = 15              # 送信上限
DEFAULT_LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "180"))

# ===== Discord Webhook =====
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")

if not DISCORD_WEBHOOK_URL:
    print("[ERROR] Set DISCORD_WEBHOOK_URL env.", file=sys.stderr)

# ===== ユーティリティ =====
def now_jst():
    return datetime.utcnow() + timedelta(hours=TZ_OFFSET)

def is_weekend(dt: datetime) -> bool:
    # 土日スキップ（日本の祝日は考慮しない）
    return dt.weekday() >= 5

def chunk_text(text: str, limit: int = 1900):
    """
    Discordのcontent制限(2000文字)を考慮して分割（余裕1900）。
    改行単位で分割。長すぎる1行は強制的に切る。
    """
    out, buf, size = [], [], 0
    for line in text.splitlines():
        if len(line) > limit:
            # 1行が超長いときは強制分割
            while len(line) > limit:
                part = line[:limit]
                line = line[limit:]
                if buf:
                    out.append("\n".join(buf))
                    buf, size = [], 0
                out.append(part)
            if line == "":
                continue
        add = len(line) + 1
        if size + add > limit:
            out.append("\n".join(buf))
            buf, size = [line], len(line) + 1
        else:
            buf.append(line)
            size += add
    if buf:
        out.append("\n".join(buf))
    return out

# ===== 日経225ティッカー =====
nikkei225_tickers = [ '4151.T','4502.T','4503.T','4506.T','4507.T','4519.T','4523.T','4568.T','4578.T','6479.T','6501.T','6503.T','6504.T','6506.T','6526.T','6594.T','6645.T','6674.T','6701.T','6702.T','6723.T','6724.T','6752.T','6753.T','6758.T','6762.T','6770.T','6841.T','6857.T','6861.T','6902.T','6920.T','6952.T','6954.T','6971.T','6976.T','6981.T','7735.T','7751.T','7752.T','8035.T','7201.T','7202.T','7203.T','7205.T','7211.T','7261.T','7267.T','7269.T','7270.T','7272.T','4543.T','4902.T','6146.T','7731.T','7733.T','7741.T','7762.T','9432.T','9433.T','9434.T','9613.T','9984.T','5831.T','7186.T','8304.T','8306.T','8308.T','8309.T','8316.T','8331.T','8354.T','8411.T','8253.T','8591.T','8697.T','8601.T','8604.T','8630.T','8725.T','8750.T','8766.T','8795.T','1332.T','2002.T','2269.T','2282.T','2501.T','2502.T','2503.T','2801.T','2802.T','2871.T','2914.T','3086.T','3092.T','3099.T','3382.T','7453.T','8233.T','8252.T','8267.T','9843.T','9983.T','2413.T','2432.T','3659.T','4307.T','4324.T','4385.T','4661.T','4689.T','4704.T','4751.T','4755.T','6098.T','6178.T','7974.T','9602.T','9735.T','9766.T','1605.T','3401.T','3402.T','3861.T','3405.T','3407.T','4004.T','4005.T','4021.T','4042.T','4043.T','4061.T','4063.T','4183.T','4188.T','4208.T','4452.T','4901.T','4911.T','6988.T','5019.T','5020.T','5101.T','5108.T','5201.T','5214.T','5233.T','5301.T','5332.T','5333.T','5401.T','5406.T','5411.T','3436.T','5706.T','5711.T','5713.T','5714.T','5801.T','5802.T','5803.T','2768.T','8001.T','8002.T','8015.T','8031.T','8053.T','8058.T','1721.T','1801.T','1802.T','1803.T','1808.T','1812.T','1925.T','1928.T','1963.T','5631.T','6103.T','6113.T','6273.T','6301.T','6302.T','6305.T','6326.T','6361.T','6367.T','6471.T','6472.T','6473.T','7004.T','7011.T','7013.T','7012.T','7832.T','7911.T','7912.T','7951.T','3289.T','8801.T','8802.T','8804.T','8830.T','9001.T','9005.T','9007.T','9008.T','9009.T','9020.T','9021.T','9022.T','9064.T','9147.T','9101.T','9104.T','9107.T','9201.T','9202.T','9301.T','9501.T','9502.T','9503.T','9531.T','9532.T' ]

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

# ======= RSI 計算ヘルパー（LINE版から踏襲） =======
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

# ===== Discord送信 =====
def discord_send_content(msg: str):
    r = requests.post(
        DISCORD_WEBHOOK_URL,
        json={"content": msg},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if r.status_code >= 300:
        raise RuntimeError(f"Discord content failed: {r.status_code} {r.text}")


def discord_send_embed(
    title: str,
    description: str | None = None,
    image_url: str | None = None,
    fields: list | None = None,
):
    embed = {"title": title, "timestamp": datetime.utcnow().isoformat() + "Z"}
    if description:
        embed["description"] = description
    if image_url:
        embed["image"] = {"url": image_url}
    if fields:
        embed["fields"] = fields

    r = requests.post(
        DISCORD_WEBHOOK_URL,
        json={"embeds": [embed]},
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    if r.status_code >= 300:
        raise RuntimeError(f"Discord embed failed: {r.status_code} {r.text}")


def discord_send_image_file(
    file_path: str,
    title: str,
    description: str | None = None,
    fields: list | None = None,
):
    """画像ファイルをWebhookに直接添付して送る（外部URL不要）"""
    embed = {"title": title, "timestamp": datetime.utcnow().isoformat() + "Z"}
    if description:
        embed["description"] = description
    if fields:
        embed["fields"] = fields

    filename = os.path.basename(file_path)
    # 添付ファイルは attachment://<filename> で参照
    embed["image"] = {"url": f"attachment://{filename}"}

    with open(file_path, "rb") as f:
        files = {"file": (filename, f, "image/png")}
        data = {"payload_json": json.dumps({"embeds": [embed]})}
        r = requests.post(DISCORD_WEBHOOK_URL, files=files, data=data, timeout=30)
        if r.status_code >= 300:
            raise RuntimeError(
                f"Discord image upload failed: {r.status_code} {r.text}"
            )


def send_long_text(msg: str):
    discord_send_content(msg)

# ===== データ取得 =====
def load_tickers():
    # 優先: 環境変数 TICKERS_CSV のCSV（Ticker/Symbol/Code列）
    csv_path = os.getenv("TICKERS_CSV")
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        col = None
        for c in df.columns:
            if c.lower() in ("ticker", "symbol", "code"):
                col = c
                break
        if col:
            tickers = [str(x).strip() for x in df[col].dropna().unique().tolist()]
            if tickers:
                return tickers

    # サンプル（あとで全銘柄に差し替えOK）
    return nikkei225_tickers

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
    # 必須カラムが揃っているか簡易チェック
    for c in ("Close", "High", "Low"):
        if c not in raw.columns.get_level_values(0):
            raise RuntimeError(f"yfinance returned missing column: {c}")
    close = raw["Close"].copy()
    high = raw["High"].copy()
    low = raw["Low"].copy()
    return raw, close, high, low

# ===== 押し目抽出（厳しい条件・LINE版踏襲） =====

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()

def _atr20(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """ATR(20) = 20日EMA(TR)。TR=max(H-L, |H-PrevC|, |L-PrevC|)"""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return _ema(tr, 20)

def _zigzag_last_swing_low(high: pd.Series, low: pd.Series, close: pd.Series,
                           thresh_pct: float, atr_mult: float) -> tuple[pd.Timestamp, float] | None:
    """
    直近window内で“意味のある谷（Swing Low）”を1つ返す（ZigZag風）
    ・直近の局所高値から閾値（% または ATR20×mult）以上下落→谷候補確定
    ・その後の上昇でピボット確定
    戻り値: (安値日, 安値) / 見つからなければ None
    """
    atr20 = _atr20(high, low, close)
    pivot_high_idx = None
    pivot_high_val = None
    last_swing_low = None

    for i in range(len(close)):
        h, l = float(high.iloc[i]), float(low.iloc[i])
        if pivot_high_val is None or h > pivot_high_val:
            pivot_high_val = h
            pivot_high_idx = i

        drop_pct = (pivot_high_val - l) / pivot_high_val if pivot_high_val else 0.0
        atr_ok = (pivot_high_val - l) >= (atr_mult * float(atr20.iloc[i] if pd.notna(atr20.iloc[i]) else 0.0))
        if drop_pct >= thresh_pct or atr_ok:
            seg = low.iloc[pivot_high_idx:i+1]
            if not seg.empty:
                j = seg.idxmin()
                last_swing_low = (j, float(low.loc[j]))
            pivot_high_val = h
            pivot_high_idx = i

    return last_swing_low

def compute_one_ticker(close_s: pd.Series, high_s: pd.Series, low_s: pd.Series, window_days=60):
    """
    新仕様：
      ① 上昇トレンド（SMA25>SMA75、両者の傾き>=0）
      ② 直近スイングロー（ZigZag風）を特定
      ③ Peak→Lowの“深さ” 下限（% または ATR20×mult）
      ④ Swing Low が SMA25 にタッチ（±SMA_TOUCH_TOL）or いったん割って即回復
      ⑤ Low からの経過日数 >= DAYS_SINCE_LOW_MIN
      ⑥ 反発率 REBOUND_MIN〜REBOUND_MAX
    満たせば dict を返す。満たさなければ None。
    """
    if close_s is None or close_s.empty:
        return None

    # 対象窓の切り出し（NaN落とし＆整列）
    close = close_s.dropna().iloc[-window_days:]
    high  = high_s.reindex(close.index).ffill()
    low   = low_s.reindex(close.index).ffill()
    if len(close) < max(30, window_days // 2):
        return None

    # トレンド：SMA25 / SMA75 と傾き
    sma25 = close.rolling(25, min_periods=25).mean()
    sma75 = close.rolling(75, min_periods=75).mean()
    if sma25.isna().all() or sma75.isna().all():
        return None
    if not (sma25.iloc[-1] > sma75.iloc[-1]):
        return None
    slope25 = sma25.iloc[-1] - (sma25.iloc[-6] if len(sma25.dropna()) >= 6 else sma25.iloc[-1])
    slope75 = sma75.iloc[-1] - (sma75.iloc[-6] if len(sma75.dropna()) >= 6 else sma75.iloc[-1])
    if slope25 <= 0 or slope75 < 0:
        return None

    # 直近スイングロー（ZigZag）
    swing = _zigzag_last_swing_low(high, low, close, DEPTH_MIN_PCT, ATR20_MULT_MIN)
    if swing is None:
        return None
    low_date, low_val = swing
    if low_date not in close.index:
        return None

    # Peak（low_date 以前の高値）
    before = close.loc[:low_date]
    if before.empty:
        return None
    peak_val = float(before.max())
    peak_date = before.idxmax()

    # 押しの深さ（%とATRの両条件）
    drop_pct = (peak_val - low_val) / peak_val if peak_val else 0.0
    if drop_pct < DEPTH_MIN_PCT:
        return None
    atr20 = _atr20(high, low, close)
    atr_ok = (peak_val - low_val) >= (ATR20_MULT_MIN * float(atr20.loc[low_date] if (low_date in atr20.index and pd.notna(atr20.loc[low_date])) else 0.0))
    if not atr_ok:
        return None

    # SMA25 タッチ（±許容 or 一時割れ→回復）
    sma25_at_low = float(sma25.loc[low_date]) if (low_date in sma25.index and pd.notna(sma25.loc[low_date])) else None
    if sma25_at_low is None:
        return None
    touch_ok = abs(low_val - sma25_at_low) / sma25_at_low <= SMA_TOUCH_TOL or (low_val <= sma25_at_low <= float(close.iloc[-1]))
    if not touch_ok:
        return None

    # 反発・経過日数
    latest = float(close.iloc[-1])
    prev   = float(close.iloc[-2]) if len(close) >= 2 else np.nan
    days_since_low = (close.index[-1] - low_date).days
    if days_since_low < DAYS_SINCE_LOW_MIN:
        return None
    rebound_pct = (latest / low_val) - 1.0
    if not (REBOUND_MIN <= rebound_pct <= REBOUND_MAX):
        return None

    # 目標値：Low以降の高値（無ければ窓内高値）
    after = close.loc[low_date:]
    target = float(after.max()) if not after.empty else float(close.max())
    expected_rise_pct = (target / latest - 1.0) * 100.0 if latest > 0 else None

    # ← ここが重要：notify が参照するキー名に完全準拠
    return {
        "Ticker": close_s.name,
        "Peak_Date": peak_date.to_pydatetime().date(),
        "Peak_High": round(peak_val, 2),
        "Pullback_Date": low_date.to_pydatetime().date(),
        "Pullback_Low": round(low_val, 2),
        "Latest_Date": close.index[-1].to_pydatetime().date(),
        "Latest_Close": round(latest, 2),
        "Prev_Close": round(prev, 2) if not np.isnan(prev) else np.nan,
        "Expected_Upper": round(target, 2),
        "Expected_Rise_%": round(expected_rise_pct, 2) if expected_rise_pct is not None else np.nan,
        # 参考情報（通知で使わないがデバッグ用）
        "Drop_From_Peak_%": round(drop_pct * 100.0, 2),
        "Rebound_From_Low_%": round(rebound_pct * 100.0, 2),
        "Days_Since_Pullback": int(days_since_low),
        "SMA25": round(float(sma25.iloc[-1]), 2) if pd.notna(sma25.iloc[-1]) else np.nan,
        "SMA75": round(float(sma75.iloc[-1]), 2) if pd.notna(sma75.iloc[-1]) else np.nan,
        "Window": int(window_days),
    }

def find_pullback_candidates(close_df: pd.DataFrame, high_df: pd.DataFrame, low_df: pd.DataFrame, window_days=30):
    rows = []
    for ticker in close_df.columns:
        res = compute_one_ticker(close_df[ticker], high_df[ticker], low_df[ticker], window_days=window_days)
        if res:
            rows.append(res)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("Expected_Rise_%", ascending=False).reset_index(drop=True)
    return df

# ===== チャート画像作成（踏襲） =====
def save_chart_image_from_raw(raw_df, ticker: str, out_dir="charts"):
    """
    raw_df: yf.download(..., group_by='column') の MultiIndex DataFrame
    """
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
    # 既存のグローバル辞書を参照。無ければ空文字。
    return {t: ticker_name_map.get(t, "") for t in tickers}

# ===== パイプライン =====
def run_pipeline():
    tickers = load_tickers()
    raw, close, high, low = fetch_market_data(tickers, lookback_days=DEFAULT_LOOKBACK_DAYS)

    # 60日・30日で抽出 → マージ（同一ティッカーは 'Expected_Rise_%' が大きい方を採用）
    rs = []
    for w in (60, 30):
        df = find_pullback_candidates(close, high, low, window_days=w)
        if not df.empty:
            df["Window"] = w
            rs.append(df)

    if not rs:
        return pd.DataFrame(), raw, {}

    cat = (
        pd.concat(rs, ignore_index=True)
          .sort_values(["Ticker", "Expected_Rise_%"], ascending=[True, False])
    )
    best = (
        cat.groupby("Ticker", as_index=False)
           .first()
           .sort_values("Expected_Rise_%", ascending=False)
           .reset_index(drop=True)
    )

    name_map = build_ticker_name_map(best["Ticker"].tolist())
    return best, raw, name_map
# ===== 通知（Discord版） =====
def notify(best_df: pd.DataFrame, raw_df, ticker_name_map: dict, top_n=TOP_N):
    if best_df is None or best_df.empty:
        discord_send_content("【押し目スクリーニング】本日は抽出なしでした。")
        return

    header = (
        f"★★★★★【押し目】★★★★★ {now_jst().strftime('%m/%d %H:%M')}\n"
        f"抽出: {len(best_df)} 銘柄（重複統合）\n"
        f"条件: {REBOUND_MAX*100:.0f}%≥反発≥{REBOUND_MIN*100:.0f}%・下落≤{DROP_MAX:.0f}%・SMA25上・期待≥{EXPECTED_RISE_MIN:.0f}%・{DAYS_SINCE_LOW_MIN}日経過\n"
        f"------------------------------"
    )
    send_long_text(header)

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

        pull_date = r.get("Pullback_Date")
        pull_str = pull_date.strftime("%m/%d") if hasattr(pull_date, "strftime") else "-"
        rsi_val = latest_rsi_from_raw(raw_df, ticker, period=14)
        rsi_str = "-" if rsi_val is None or not np.isfinite(rsi_val) else f"{rsi_val:.0f}"

        # テキスト 5行（content）
        line1 = f"{ticker} {name}".rstrip()
        line2 = f"底日 {pull_str}"
        line3 = f"↗ {fpct(rise_p)}   🎯 上 {fnum(upper)}   下 {fnum(low)}"
        line4 = f"今 {fnum(latest)}   🎯 期待 {fnum(expect_amt)}  RSI {rsi_str}"
        line5 = f"変動率 {fpct_signed(chg_pct)}   底値比較 {fpct_signed(bot_pct)}"
        msg = "\n".join([line1, line2, line3, line4, line5])
        send_long_text(msg)

        # チャート画像（Embed or 添付）
        img_path = save_chart_image_from_raw(raw_df, ticker, out_dir="charts")
        title = f"{ticker} {name}".strip()
        desc = f"Window: best / 期待上昇 {fpct(rise_p)}"
        fields = [
            {"name": "Pullback", "value": f"{pull_str}",     "inline": True},
            {"name": "Latest",   "value": f"{fnum(latest)}", "inline": True},
            {"name": "Target",   "value": f"{fnum(upper)}",  "inline": True},
        ]

        if img_path:
            if PUBLIC_BASE_URL:
                public_url = f"{PUBLIC_BASE_URL}/{os.path.basename(img_path)}"
                discord_send_embed(
                    title=title,
                    description=desc,
                    image_url=public_url,
                    fields=fields,
                )
            else:
                discord_send_image_file(
                    file_path=img_path,
                    title=title,
                    description=desc,
                    fields=fields,
                )
        else:
            discord_send_embed(
                title=title,
                description=desc,
                fields=fields,
            )


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
