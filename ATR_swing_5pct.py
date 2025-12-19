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

# ★ 抽出ロジック用パラメータ（ここだけロジック関連）
# --- A（バランス型）：+3% 指値が刺さりやすい “高値側のクセ”も加味 ---
ATR_MIN_PCT = 4.5               # ATR20_pct >= 4.5%
UPPER_ATR_MIN_PCT = 2.0         # Upper_ATR20_pct >= 2.0%
HIGH_REACH_RATIO_MIN = 0.50     # HighReachRatio20(3%) >= 50%（=0.50）
WEEK_RET_MIN = 0.0              # 5営業日前比リターン >= 0
REACH_PCT = 3.0                 # 高値到達判定の閾値（前日終値×(1+3%)）

# チャート設定（GC_MACD_BB と同じ）
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


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


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


def send_long_text(msg: str):
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


# ===== yfinance データ取得 =====
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


# ===== 抽出ロジック：A（バランス型） =====
def latest_atr_swing_from_raw(raw_df: pd.DataFrame, ticker: str):
    """
    ロジック（A: バランス型）：
      - ATR20_pct >= ATR_MIN_PCT
      - Upper_ATR20_pct >= UPPER_ATR_MIN_PCT
      - HighReachRatio20(3%) >= HIGH_REACH_RATIO_MIN
         * HighReach = (当日高値 >= 前日終値 * (1 + REACH_PCT/100))
         * Ratio20 = 過去20営業日の HighReach 成立比率
      - 5営業日前比リターン >= WEEK_RET_MIN
    を最新日で判定し、指標を返す。
    """

    try:
        # --- 終値・高値・安値シリーズを取り出す ---
        if isinstance(raw_df.columns, pd.MultiIndex):
            close = raw_df[("Close", ticker)].dropna()
            high = raw_df[("High", ticker)].reindex(close.index)
            low = raw_df[("Low", ticker)].reindex(close.index)
        else:
            close = raw_df["Close"].dropna()
            high = raw_df["High"].reindex(close.index)
            low = raw_df["Low"].reindex(close.index)

        # 必要期間（ATR20 + Reach20 + 週足判定5）を考慮して余裕を持たせる
        if len(close) < 50:
            return None

        prev_close = close.shift(1)

        # --- ATR20（True Rangeベース）---
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr20 = tr.rolling(window=20, min_periods=20).mean()
        atr20_pct = atr20 / close * 100.0

        # --- Upper_ATR20（高値側の動き：上ヒゲ/ギャップ上方向を重視）---
        upper_tr1 = high - close
        upper_tr2 = (high - prev_close).abs()
        upper_tr = pd.concat([upper_tr1, upper_tr2], axis=1).max(axis=1)

        upper_atr20 = upper_tr.rolling(window=20, min_periods=20).mean()
        upper_atr20_pct = upper_atr20 / close * 100.0

        # --- HighReachRatio20(3%) ---
        target_mult = 1.0 + (REACH_PCT / 100.0)
        reach = (high >= prev_close * target_mult).astype(float)
        high_reach_ratio20 = reach.rolling(window=20, min_periods=20).mean()

        # --- 週足がマイナスじゃない（5営業日前比で非マイナス） ---
        close_5ago = close.shift(5)
        week_ret = close / close_5ago - 1.0

        # 最新値
        atr_v = atr20_pct.iloc[-1]
        upper_atr_v = upper_atr20_pct.iloc[-1]
        reach_v = high_reach_ratio20.iloc[-1]
        week_v = week_ret.iloc[-1]

        # NaNガード
        if np.isnan(atr_v) or np.isnan(upper_atr_v) or np.isnan(reach_v) or np.isnan(week_v):
            return None

        atr_ok = atr_v >= ATR_MIN_PCT
        upper_ok = upper_atr_v >= UPPER_ATR_MIN_PCT
        reach_ok = reach_v >= HIGH_REACH_RATIO_MIN
        week_ok = week_v >= WEEK_RET_MIN

        ok = bool(atr_ok and upper_ok and reach_ok and week_ok)

        return {
            "ok": ok,
            "ATR20_pct": float(atr_v),
            "Upper_ATR20_pct": float(upper_atr_v),
            "HighReachRatio20": float(reach_v),
            "WeekRet": float(week_v),
        }

    except Exception:
        return None


# ===== チャート生成（GC_MACD_BBと同じ） =====
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


# ===== シグナル判定 & 通知 =====
def screen_atr_swing(raw_df: pd.DataFrame, tickers):
    """
    A（バランス型）:
      ATR20_pct >= 4.5%
      Upper_ATR20_pct >= 2.0%
      HighReachRatio20(3%) >= 50%
      5営業日前比リターン >= 0
    """
    rows = []
    for t in tickers:
        res = latest_atr_swing_from_raw(raw_df, t)
        if res is None:
            continue
        if not res.get("ok", False):
            continue
        rows.append({
            "Ticker": t,
            "ATR20_pct": res["ATR20_pct"],
            "Upper_ATR20_pct": res["Upper_ATR20_pct"],
            "HighReachRatio20": res["HighReachRatio20"],
            "WeekRet": res["WeekRet"],
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # ATRの大きさ優先でソート（より激しく動く銘柄を上に）
    df = df.sort_values("ATR20_pct", ascending=False).reset_index(drop=True)
    return df


def notify(df: pd.DataFrame, raw_df: pd.DataFrame):
    # --- テキスト部 ---
    title = "【ATRスイング候補（A: 指値+3%向け / バランス型）】"

    cond_line = (
        f"条件: ATR20%>={ATR_MIN_PCT:.1f}% / UpperATR20%>={UPPER_ATR_MIN_PCT:.1f}% "
        f"/ HighReach20({REACH_PCT:.0f}%)>={HIGH_REACH_RATIO_MIN*100:.0f}% "
        f"/ WeekRet(5d)>={WEEK_RET_MIN:.1%}"
    )

    if df is None or df.empty:
        msg = (
            f"{title} {now_jst():%m/%d %H:%M}\n"
            f"{cond_line}\n"
            f"該当銘柄はありませんでした。"
        )
        send_long_text(msg)
        return

    header = (
        f"{title} {now_jst():%m/%d %H:%M}\n"
        f"{cond_line}\n"
        f"抽出: {len(df)} 銘柄\n"
        f"------------------------------"
    )

    lines = [header]

    def fp(v, digits=2):
        try:
            return f"{float(v):.{digits}f}"
        except Exception:
            return "-"

    for _, r in df.iterrows():
        t = r["Ticker"]
        name = TICKER_NAME_MAP.get(t, "")
        atr = fp(r["ATR20_pct"], 2)
        uatr = fp(r["Upper_ATR20_pct"], 2)
        reach = fp(100.0 * r["HighReachRatio20"], 1)
        wret = fp(100 * r["WeekRet"], 2) if r["WeekRet"] is not None else "-"
        line = f"{t:<8} {name:<8}  ATR20%:{atr:>6}  UpperATR%:{uatr:>6}  Reach20%:{reach:>6}  WeekRet%:{wret:>6}"
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
        title = f"{t} {name}".strip()
        desc = (
            f"ATR20%:{fp(r['ATR20_pct'],2)} "
            f"UpperATR%:{fp(r['Upper_ATR20_pct'],2)} "
            f"Reach20%:{fp(100.0 * r['HighReachRatio20'],1)} "
            f"WeekRet%:{fp(100 * r['WeekRet'],2) if r['WeekRet'] is not None else '-'}"
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
    df = screen_atr_swing(raw, tickers)
    notify(df, raw)


if __name__ == "__main__":
    main()
