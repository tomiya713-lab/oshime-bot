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

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
DISCORD_ENABLED = bool(DISCORD_WEBHOOK_URL)

FORCE_RUN = os.getenv("FORCE_RUN", "0") == "1"

# 判定用パラメータ（※抽出条件に関わるため今回ロジックに合わせて置換）
SMA_SHORT = 5
SMA_LONG = 25

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

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


def send_long_text(msg: str):
    for part in chunk_text(msg):
        discord_send_content(part)


# ===== ティッカー一覧 =====

# フォールバックとして日経225ティッカー
NIKKEI225_TICKERS = ['4151.T','4502.T','4503.T','4506.T','4507.T','4519.T','4523.T','4543.T','4568.T','4578.T',
'4661.T','4689.T','4704.T','4751.T','4755.T','4901.T','4902.T','4911.T','4922.T','4931.T',
'5019.T','5020.T','5101.T','5108.T','5201.T','5202.T','5214.T','5232.T','5233.T','5301.T',
'5332.T','5333.T','5401.T','5406.T','5411.T','5541.T','5631.T','5706.T','5707.T','5711.T',
'5713.T','5801.T','5802.T','5803.T','5901.T','6098.T','6103.T','6113.T','6178.T','6273.T',
'6301.T','6361.T','6367.T','6471.T','6472.T','6473.T','6479.T','6501.T','6503.T','6504.T',
'6506.T','6508.T','6645.T','6674.T','6701.T','6702.T','6723.T','6724.T','6752.T','6762.T',
'6770.T','6841.T','6857.T','6861.T','6902.T','6952.T','6954.T','6971.T','6976.T','6981.T',
'7011.T','7012.T','7013.T','7014.T','6971.T','7186.T','7201.T','7202.T','7203.T','7205.T',
'7211.T','7267.T','7269.T','7270.T','7272.T','7733.T','7741.T','7751.T','7752.T','7762.T',
'7832.T','7911.T','7912.T','7951.T','7974.T','8001.T','8002.T','8015.T','8031.T','8035.T',
'8053.T','8058.T','8306.T','8316.T','8411.T','8601.T','8604.T','8630.T','8725.T','8750.T',
'8766.T','8801.T','8802.T','8830.T','9001.T','9005.T','9007.T','9008.T','9009.T','9020.T',
'9021.T','9022.T','9024.T','9064.T','9101.T','9104.T','9107.T','9147.T','9201.T','9202.T',
'9301.T','9432.T','9433.T','9434.T','9501.T','9502.T','9503.T','9531.T','9532.T','9602.T',
'9735.T','9766.T','9843.T','9983.T','9984.T','4063.T','2802.T','3402.T','3407.T','3659.T',
'3861.T','4004.T','4005.T','4021.T','4042.T','4183.T','4188.T','4208.T','4452.T','4502.T',
'5108.T','5401.T','5406.T','5711.T','6471.T','6501.T','6503.T','6762.T','6902.T','6952.T',
'6971.T','7201.T','7202.T','7203.T','7267.T','7269.T','8035.T','8058.T','8306.T','8316.T',
'8411.T','8601.T','8725.T','8766.T','8801.T','8802.T','9020.T','9022.T','9101.T','9104.T',
'9107.T','9432.T','9433.T','9434.T','9501.T','9503.T','9983.T','9984.T','6146.T','7731.T']

TICKER_NAME_MAP = {
    "4151.T": "協和キリン", "4502.T": "武田", "4503.T": "アステラス", "4506.T": "住友ファーマ",
    "4507.T": "塩野義", "4519.T": "中外製薬", "4523.T": "エーザイ", "4543.T": "テルモ",
    "4568.T": "第一三共", "4578.T": "大塚HD", "4661.T": "オリコン", "4689.T": "ZHD",
    "4704.T": "トレンドM", "4751.T": "サイバー", "4755.T": "楽天G", "4901.T": "富士フイルム",
    "4902.T": "コニカミノル", "4911.T": "資生堂", "4922.T": "コーセー", "4931.T": "新日本製薬",
    "5019.T": "出光", "5020.T": "ENEOS", "5101.T": "横浜ゴム", "5108.T": "ブリヂストン",
    "5201.T": "AGC", "5202.T": "日本板硝子", "5214.T": "日電硝子", "5232.T": "住友大阪",
    "5233.T": "太平洋セメ", "5301.T": "東海カ", "5332.T": "TOTO", "5333.T": "日本ガイシ",
    "5401.T": "日本製鉄", "5406.T": "神戸鋼", "5411.T": "JFE", "5541.T": "大平金",
    "5631.T": "日製鋼", "5706.T": "三井金", "5707.T": "東邦鉛", "5711.T": "三菱マ",
    "5713.T": "住友鉱", "5801.T": "古河電工", "5802.T": "住友電工", "5803.T": "フジクラ",
    "5901.T": "東プレ", "6098.T": "リクルート", "6103.T": "オークマ", "6113.T": "アマダ",
    "6178.T": "日本郵政", "6273.T": "SMC", "6301.T": "コマツ", "6361.T": "荏原",
    "6367.T": "ダイキン", "6471.T": "日精工", "6472.T": "NTN", "6473.T": "ジェイテクト",
    "6479.T": "ミネベア", "6501.T": "日立", "6503.T": "三菱電", "6504.T": "富士電機",
    "6506.T": "安川電", "6508.T": "明電舎", "6645.T": "オムロン", "6674.T": "GSユアサ",
    "6701.T": "NEC", "6702.T": "富士通", "6723.T": "ルネサス", "6724.T": "セイコーE",
    "6752.T": "パナ", "6762.T": "TDK", "6770.T": "アルプス", "6841.T": "横河電機",
    "6857.T": "アドテスト", "6861.T": "キーエンス", "6902.T": "デンソー", "6952.T": "カシオ",
    "6954.T": "ファナック", "6971.T": "京セラ", "6976.T": "太陽誘電", "6981.T": "村田",
    "7011.T": "三菱重工", "7012.T": "川重", "7013.T": "IHI", "7014.T": "名村造船",
    "7186.T": "コンコルディア", "7201.T": "日産", "7202.T": "いすゞ", "7203.T": "トヨタ",
    "7205.T": "日野", "7211.T": "三菱自", "7267.T": "ホンダ", "7269.T": "スズキ",
    "7270.T": "SUBARU", "7272.T": "ヤマハ発", "7733.T": "オリンパス", "7741.T": "HOYA",
    "7751.T": "キヤノン", "7752.T": "リコー", "7762.T": "シチズン", "7832.T": "バンナム",
    "7911.T": "TOPPAN", "7912.T": "大日印", "7951.T": "ヤマハ", "7974.T": "任天堂",
    "8001.T": "伊藤忠", "8002.T": "丸紅", "8015.T": "豊通", "8031.T": "三井物産",
    "8035.T": "東エレク", "8053.T": "住友商", "8058.T": "三菱商", "8306.T": "三菱UFJ",
    "8316.T": "三井住友", "8411.T": "みずほ", "8601.T": "大和", "8604.T": "野村",
    "8630.T": "SOMPO", "8725.T": "MS&AD", "8750.T": "第一生命HD", "8766.T": "東京海上",
    "8801.T": "三井不", "8802.T": "三菱地", "8830.T": "住友不", "9001.T": "東武",
    "9005.T": "東急", "9007.T": "小田急", "9008.T": "京王", "9009.T": "京成",
    "9020.T": "JR東日本", "9021.T": "JR西日本", "9022.T": "JR東海", "9024.T": "西武HD",
    "9064.T": "ヤマトHD", "9101.T": "日本郵船", "9104.T": "商船三井", "9107.T": "川崎汽船",
    "9147.T": "NXHD", "9201.T": "JAL", "9202.T": "ANAHD", "9301.T": "三菱倉庫",
    "9432.T": "NTT", "9433.T": "KDDI", "9434.T": "ソフトバンク", "9501.T": "東電HD",
    "9502.T": "中部電", "9503.T": "関西電", "9531.T": "東ガス", "9532.T": "大阪ガス",
    "9602.T": "東宝", "6963.T": "ローム", "9735.T": "セコム", "9766.T": "コナミG",
    "9843.T": "ニトリHD", "9983.T": "ファーストリテ", "9984.T": "ソフトバンクG",
}


def load_tickers():
    # CSV入力は使わない運用（常に日経225を対象）
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


# ===== 指標計算 =====
def latest_gc_macd_bb_from_raw(raw_df: pd.DataFrame, ticker: str):
    """
    最終版ロジック（GC_short + MACD強気 + 週足BB上限ブレイク週）を
    最新日で判定し、指標値も返す。
    """
    try:
        if isinstance(raw_df.columns, pd.MultiIndex):
            close = raw_df[("Close", ticker)].dropna()
        else:
            close = raw_df["Close"].dropna()
        if len(close) < max(SMA_LONG, MACD_SLOW, WBB_PERIOD_WEEKS * 5) + 5:
            return None

        # --- 日足SMA ---
        sma_short = close.rolling(window=SMA_SHORT, min_periods=SMA_SHORT).mean()
        sma_long = close.rolling(window=SMA_LONG, min_periods=SMA_LONG).mean()

        # GC_short（今日クロスしたか）
        gc_today = (sma_short.iloc[-1] > sma_long.iloc[-1]) and \
                   (sma_short.iloc[-2] <= sma_long.iloc[-2])

        # --- MACD (12,26,9) ---
        ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()

        macd_bullish = (macd_line.iloc[-1] > 0) and \
                       (macd_line.iloc[-1] > macd_signal.iloc[-1])

        # --- 週足BB上限 (W-FRIで週足化) ---
        daily = close.to_frame("Close")
        weekly_close = daily.resample("W-FRI").last()["Close"].dropna()

        if len(weekly_close) < WBB_PERIOD_WEEKS + 2:
            return None

        w_ma20 = weekly_close.rolling(window=WBB_PERIOD_WEEKS, min_periods=WBB_PERIOD_WEEKS).mean()
        w_std20 = weekly_close.rolling(window=WBB_PERIOD_WEEKS, min_periods=WBB_PERIOD_WEEKS).std()
        w_upper = w_ma20 + WBB_SIGMA * w_std20

        wbb_upper_touch = False
        if not pd.isna(w_upper.iloc[-1]):
            wbb_upper_touch = (weekly_close.iloc[-1] >= w_upper.iloc[-1])

        ok = gc_today and macd_bullish and wbb_upper_touch

        return {
            # 判定フラグ
            "ok": bool(ok),
            "gc_today": bool(gc_today),
            "macd_bullish": bool(macd_bullish),
            "wbb_upper_touch": bool(wbb_upper_touch),

            # 指標値（最新）
            "SMA5": float(sma_short.iloc[-1]),
            "SMA25": float(sma_long.iloc[-1]),
            "SMA_gap_pct": float((sma_short.iloc[-1] / sma_long.iloc[-1] - 1.0) * 100.0) if float(sma_long.iloc[-1]) != 0.0 else np.nan,
            "MACD": float(macd_line.iloc[-1]),
            "MACD_signal": float(macd_signal.iloc[-1]),
            "WClose": float(weekly_close.iloc[-1]) if len(weekly_close) > 0 else np.nan,
            "WBB_upper": float(w_upper.iloc[-1]) if len(w_upper) > 0 else np.nan,
        }
    except Exception:
        return None


# ===== チャート生成 =====
def save_chart_image_from_raw(raw_df: pd.DataFrame, ticker: str, out_dir: str = CHART_OUT_DIR):
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

        use = use.dropna()
        use = use.tail(CHART_LOOKBACK_DAYS)

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


# ===== 全銘柄CSV（閾値調整用） =====
def compute_all_metrics(raw_df: pd.DataFrame, tickers):
    """全銘柄分の指標を計算してCSV出力用のDataFrameを返す（抽出条件で落とさない）。"""
    rows = []
    for t in tickers:
        res = latest_gc_macd_bb_from_raw(raw_df, t)
        if res is None:
            rows.append({
                "Ticker": t,
                "Name": TICKER_NAME_MAP.get(t, ""),
                "ok": False,
                "gc_today": False,
                "macd_bullish": False,
                "wbb_upper_touch": False,
                "SMA5": np.nan,
                "SMA25": np.nan,
                "SMA_gap_pct": np.nan,
                "MACD": np.nan,
                "MACD_signal": np.nan,
                "WClose": np.nan,
                "WBB_upper": np.nan,
                "Reason": "insufficient_or_error",
            })
            continue

        rows.append({
            "Ticker": t,
            "Name": TICKER_NAME_MAP.get(t, ""),
            "ok": bool(res.get("ok", False)),
            "gc_today": bool(res.get("gc_today", False)),
            "macd_bullish": bool(res.get("macd_bullish", False)),
            "wbb_upper_touch": bool(res.get("wbb_upper_touch", False)),
            "SMA5": res.get("SMA5", np.nan),
            "SMA25": res.get("SMA25", np.nan),
            "SMA_gap_pct": res.get("SMA_gap_pct", np.nan),
            "MACD": res.get("MACD", np.nan),
            "MACD_signal": res.get("MACD_signal", np.nan),
            "WClose": res.get("WClose", np.nan),
            "WBB_upper": res.get("WBB_upper", np.nan),
            "Reason": "",
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # ok優先 → MACD強い順 → SMA乖離順
    df = df.sort_values(["ok", "MACD", "SMA_gap_pct"], ascending=[False, False, False]).reset_index(drop=True)
    return df


# ===== シグナル判定 & 通知 =====
def screen_gc_macd_bb(raw_df: pd.DataFrame, tickers):
    """
    最終版（GC_short + MACD強気 + 週足BB上限ブレイク週）を満たすティッカーを抽出。
    """
    rows = []
    for t in tickers:
        res = latest_gc_macd_bb_from_raw(raw_df, t)
        if res is None:
            continue
        if not res.get("ok", False):
            continue
        rows.append({
            "Ticker": t,
            "SMA5": res["SMA5"],
            "SMA25": res["SMA25"],
            "MACD": res["MACD"],
            "MACD_signal": res["MACD_signal"],
            "WBB_upper_touch": res["wbb_upper_touch"],
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # MACDが強い順→クロスの明確さを見るためSMA乖離順でソート
    df["SMA_gap_pct"] = (df["SMA5"] / df["SMA25"] - 1.0) * 100.0
    df = df.sort_values(["MACD", "SMA_gap_pct"], ascending=[False, False]).reset_index(drop=True)
    return df


def fp(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "-"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def notify(df: pd.DataFrame, raw_df: pd.DataFrame):
    title = f"【GC_short + MACD強気 + 週足BB上限ブレイク週】 {now_jst():%m/%d %H:%M}"

    if df is None or df.empty:
        discord_send_content(f"{title}\n対象なし")
        return

    lines = [title, f"候補：{len(df)}件", ""]
    for _, r in df.iterrows():
        t = r["Ticker"]
        name = TICKER_NAME_MAP.get(t, "")
        lines.append(
            f"{t} {name} | SMA5={fp(r['SMA5'],2)} SMA25={fp(r['SMA25'],2)} "
            f"MACD={fp(r['MACD'],3)} Sig={fp(r['MACD_signal'],3)}"
        )

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
        desc = f"SMA5:{fp(r['SMA5'],2)} SMA25:{fp(r['SMA25'],2)} MACD:{fp(r['MACD'],3)} Sig:{fp(r['MACD_signal'],3)}"

        img_path = save_chart_image_from_raw(raw_df, t, out_dir=CHART_OUT_DIR)
        if img_path:
            discord_send_image_file(img_path, title=title, description=desc)


def main():
    now = now_jst()
    if not FORCE_RUN and is_weekend(now):
        print(f"[SKIP] {now:%F %R} 週末のためスキップ（FORCE_RUN=1で強制実行）")
        return

    tickers = load_tickers()
    raw = fetch_market_data(tickers, lookback_days=LOOKBACK_DAYS)

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

    df = screen_gc_macd_bb(raw, tickers)
    notify(df, raw)


if __name__ == "__main__":
    main()
