# -*- coding: utf-8 -*-
# main3.py — 押し目抽出 → Discord Webhook に「テキスト（分割）→ チャート画像(Embed)」連続送信
# 依存: pandas, numpy, yfinance, mplfinance, requests
#
# 環境変数:
#   DISCORD_WEBHOOK_URL  (必須)
#   PUBLIC_BASE_URL      (任意; 例: https://<user>.github.io/charts など / 末尾スラ無しでもOK)
#   （任意）FORCE_RUN=1 で週末スキップ無効化
#   （任意）TICKERS_CSV=./tickers.csv  (Ticker列/Code列/Symbol列を含むCSV)
#   （任意）LOOKBACK_DAYS=180
#

import os
import sys
import math
from datetime import datetime, timedelta
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import mplfinance as mpf

# ===== 設定（必要に応じて変更） =====
TZ_OFFSET = 9  # JST
REBOUND_MIN = 1.0       # 反発率 >= 1%
REBOUND_MAX = 4.0       # 反発率 <= 4%
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
    """
    contentでシンプルに送信。2000文字制限に合わせて事前分割。
    """
    headers = {"Content-Type": "application/json"}
    for part in chunk_text(msg, limit=1900):
        payload = {"content": part}
        r = requests.post(DISCORD_WEBHOOK_URL, json=payload, headers=headers, timeout=20)
        if r.status_code >= 300:
            raise RuntimeError(f"Discord send failed: {r.status_code} {r.text}")

def discord_send_embed(title: str, description: str | None = None, image_url: str | None = None, fields: list | None = None):
    """
    Embedで送信（チャート画像URLをimageに表示）。
    """
    embed = {"title": title, "timestamp": datetime.utcnow().isoformat() + "Z"}
    if description:
        # descriptionも2000文字制限あるが、ここでは短文想定
        embed["description"] = description
    if image_url:
        embed["image"] = {"url": image_url}
    if fields:
        embed["fields"] = fields

    payload = {"embeds": [embed]}
    headers = {"Content-Type": "application/json"}
    r = requests.post(DISCORD_WEBHOOK_URL, json=payload, headers=headers, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"Discord embed failed: {r.status_code} {r.text}")

def send_long_text(msg: str):
    # 長文を自動分割して送る（content）
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
    return nikkei225_tickers

def fetch_market_data(tickers, lookback_days=DEFAULT_LOOKBACK_DAYS):
    """
    Colabの安定パターンを移植：
    - まとめてdownload（threads=False）
    - MultiIndex列から stack(dropna=False) で縦持ち化
    - 列を揃えてから pivot（横展開）
    - Closeを“分析用の終値”としてAdj_Close列名に統一
    """
    end_dt = (now_jst().date() + timedelta(days=1)).isoformat()
    start_dt = (now_jst().date() - timedelta(days=lookback_days)).isoformat()

    # まとめてDL（ここをColabと同じ方針に）
    raw = yf.download(
        tickers=tickers,
        start=start_dt,
        end=end_dt,
        interval="1d",
        auto_adjust=False,   # Colabと合わせる（Close=通常の終値）
        progress=False,
        threads=False        # ★安定優先
        # group_by は明示しない（デフォルトで column）
    )

    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned empty frame for all tickers.")

    # --- 縦持ち → 結合（Colabのやり方） ---
    # ※ dropna=False で欠損も保持し、後段で必要列だけを安全に組み立て
    try:
        s_close = raw["Close"].stack(dropna=False)
        s_open  = raw["Open" ].stack(dropna=False)
        s_high  = raw["High" ].stack(dropna=False)
        s_low   = raw["Low"  ].stack(dropna=False)
        s_vol   = raw["Volume"].stack(dropna=False)
    except Exception as e:
        raise RuntimeError(f"stack failed: {e}")

    vdf = pd.concat(
        {
            "Adj_Close": s_close,    # 分析側は“終値”としてこの列名に寄せる（Colab準拠）
            "Open":      s_open,
            "high":      s_high,
            "low":       s_low,
            "volume":    s_vol,
        },
        axis=1
    ).reset_index()
    vdf.columns = ["Date", "Ticker", "Adj_Close", "Open", "high", "low", "volume"]

    # --- 横展開：分析用DataFrame（close_df / high_df / low_df） ---
    vdf["Date"] = pd.to_datetime(vdf["Date"])
    vdf = vdf.sort_values(["Ticker", "Date"])

    close_df = vdf.pivot(index="Date", columns="Ticker", values="Adj_Close").sort_index()
    high_df  = vdf.pivot(index="Date", columns="Ticker", values="high").sort_index()
    low_df   = vdf.pivot(index="Date", columns="Ticker", values="low").sort_index()

    # main3の他処理と互換にするため raw も返す（チャート生成用）
    # チャートは (Open,High,Low,Close,Volume) を要求するため、簡易リビルド
    # -> 生の raw をそのまま返してOK（MultiIndexの (field,ticker) 構造）
    #    以降の save_chart_image_from_raw で参照できます
    return raw, close_df, high_df, low_df



# ===== 押し目抽出（厳しい条件・LINE版踏襲） =====
def rolling_sma(series: pd.Series, window=SMA_WINDOW):
    return series.rolling(window, min_periods=window).mean()

def compute_one_ticker(close_s: pd.Series, high_s: pd.Series, low_s: pd.Series, window_days=30):
    try:
        close_s = close_s.dropna()
        high_s = high_s.reindex_like(close_s).dropna()
        low_s  = low_s.reindex_like(close_s).dropna()
        if len(close_s) < max(SMA_WINDOW, window_days) + 2:
            return None

        # 対象期間
        look = close_s.iloc[-window_days:]
        look_high = high_s.loc[look.index]
        look_low  = low_s.loc[look.index]
        if look_high.empty or look_low.empty:
            return None

        # ピーク（期間内の最高値）— 最終行（当日足）を探索から除外
        look_high_use = look_high.iloc[:-1] if len(look_high) > 1 else look_high
        if look_high_use.empty:
            return None
        peak_idx = look_high_use.idxmax()
        peak_val = float(look_high_use.loc[peak_idx])

        # ピーク後の最安値（なければ除外）
        after_peak = look_low.loc[look_low.index > peak_idx]
        if after_peak.empty:
            return None
        pull_idx = after_peak.idxmin()
        pull_val = float(after_peak.loc[pull_idx])


        latest_idx = close_s.index[-1]
        latest_val = float(close_s.iloc[-1])
        prev_val = float(close_s.iloc[-2]) if len(close_s) >= 2 else np.nan

        sma25 = float(rolling_sma(close_s).iloc[-1]) if len(close_s) >= SMA_WINDOW else np.nan

        rebound_pct = (latest_val / pull_val - 1.0) * 100.0
        drop_pct = (1.0 - latest_val / peak_val) * 100.0
        expected_upper = peak_val
        expected_rise_pct = (expected_upper / latest_val - 1.0) * 100.0
        days_since_pull = (close_s.index.get_loc(latest_idx) - close_s.index.get_loc(pull_idx))

        conds = [
            rebound_pct >= REBOUND_MIN,
            rebound_pct <= REBOUND_MAX,
            drop_pct <= DROP_MAX,
            days_since_pull >= DAYS_SINCE_MIN,
            not math.isnan(sma25) and latest_val >= sma25,
            expected_rise_pct >= EXPECTED_RISE_MIN,
            latest_val >= pull_val,
        ]
        if not all(conds):
            return None

        return {
            "Ticker": close_s.name,
            "Peak_Date": peak_idx.date(),
            "Peak_High": round(peak_val, 2),
            "Pullback_Date": pull_idx.date(),
            "Pullback_Low": round(pull_val, 2),
            "Latest_Date": latest_idx.date(),
            "Latest_Close": round(latest_val, 2),
            "Prev_Close": round(prev_val, 2) if not np.isnan(prev_val) else np.nan,
            "Return_%": round(expected_rise_pct, 2),
            "Rebound_From_Low_%": round(rebound_pct, 2),
            "Drop_From_Peak_%": round(drop_pct, 2),
            "Days_Since_Pullback": int(days_since_pull),
            "SMA25": round(sma25, 2) if not math.isnan(sma25) else np.nan,
            "Expected_Upper": round(expected_upper, 2),
            "Expected_Rise_%": round(expected_rise_pct, 2),
        }
    except Exception as e:
        print(f"[WARN] compute_one_ticker failed for {close_s.name}: {e}", file=sys.stderr)
        return None

def find_pullback_candidates(close_df: pd.DataFrame, high_df: pd.DataFrame, low_df: pd.DataFrame, window_days=30):
    rows = []
    for ticker in close_df.columns:
        res = compute_one_ticker(close_df[ticker], high_df[ticker], low_df[ticker], window_days=window_days)
        if res:
            rows.append(res)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("Return_%", ascending=False).reset_index(drop=True)
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

    # 30日・14日で抽出 → マージ（同一ティッカーは 'Return_%' が大きい方を採用）
    rs = []
    for w in (30, 14):
        df = find_pullback_candidates(close, high, low, window_days=w)
        if not df.empty:
            df["Window"] = w
            rs.append(df)

    if not rs:
        return pd.DataFrame(), raw, {}

    cat = pd.concat(rs, ignore_index=True).sort_values(["Ticker", "Return_%"], ascending=[True, False])
    best = cat.groupby("Ticker", as_index=False).first().sort_values("Return_%", ascending=False).reset_index(drop=True)
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
        f"条件: {REBOUND_MAX:.0f}%≥反発≥{REBOUND_MIN:.0f}%・下落≤{DROP_MAX:.0f}%・SMA25上・期待≥{EXPECTED_RISE_MIN:.0f}%・{DAYS_SINCE_MIN}日経過\n"
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

        # チャート画像（Embed）
        img_path = save_chart_image_from_raw(raw_df, ticker, out_dir="charts")
        if img_path and PUBLIC_BASE_URL:
            public_url = f"{PUBLIC_BASE_URL}/{os.path.basename(img_path)}"
            title = f"{ticker} {name}".strip()
            desc = f"Window: best / 期待上昇 {fpct(rise_p)}"
            fields = [
                {"name": "Pullback", "value": f"{pull_str}", "inline": True},
                {"name": "Latest", "value": f"{fnum(latest)}", "inline": True},
                {"name": "Target", "value": f"{fnum(upper)}", "inline": True},
            ]
            discord_send_embed(title=title, description=desc, image_url=public_url, fields=fields)

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


