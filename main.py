# main.py
import os, requests, numpy as np, pandas as pd, yfinance as yf, jpholiday
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Tokyo")

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

# ===== LINE送信 =====
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
LINE_USER_ID = os.environ.get("LINE_USER_ID", "")  # 自分宛Push

def line_send(text: str, to_user_id: str | None = None):
    assert LINE_CHANNEL_ACCESS_TOKEN, "LINE_CHANNEL_ACCESS_TOKEN is missing"
    headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}", "Content-Type": "application/json"}
    if to_user_id is None:
        to_user_id = LINE_USER_ID
    if to_user_id:
        url = "https://api.line.me/v2/bot/message/push"
        payload = {"to": to_user_id, "messages": [{"type": "text", "text": text}]}
    else:
        url = "https://api.line.me/v2/bot/message/broadcast"
        payload = {"messages": [{"type": "text", "text": text}]}
    r = requests.post(url, headers=headers, json=payload, timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"LINE send failed: {r.status_code} {r.text}")

def send_long_text(text: str, chunk=900):
    for i in range(0, len(text), chunk):
        line_send(text[i:i+chunk])

# ===== 押し目ロジック（CSVなし版） =====
def fetch_market_data(tickers, lookback_days=180):
    end_dt = datetime.now(tz=TZ).date() + timedelta(days=1)  # 翌日までで欠損回避
    start_dt = end_dt - timedelta(days=lookback_days)
    data = yf.download(
        tickers,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    close = data["Close"]; high = data["High"]; low = data["Low"]
    return close, high, low

def compute_sma(df_close, window=25):
    return df_close.rolling(window=window, min_periods=1).mean()

def find_pullback_candidates(close, high, low, window_days=30):
    """
    改良版押し目ルール＋Prev_Close（前日終値）を追加
    """
    results = []
    sma25 = compute_sma(close, 25)
    for ticker in close.columns:
        s_close = close[ticker].dropna()
        if len(s_close) < window_days + 5:
            continue
        s_high  = high[ticker].reindex(s_close.index).ffill()
        s_low   = low[ticker].reindex(s_close.index).ffill()
        s_sma25 = sma25[ticker].reindex(s_close.index)

        w_close = s_close.iloc[-window_days:]
        w_high  = s_high.iloc[-window_days:]
        w_low   = s_low.iloc[-window_days:]
        w_sma25 = s_sma25.iloc[-window_days:]

        latest_close = float(w_close.iloc[-1])
        prev_close   = float(w_close.iloc[-2]) if len(w_close) >= 2 else np.nan
        latest_sma25 = float(w_sma25.iloc[-1])

        peak_pos  = int(np.argmax(w_high.values))
        peak_high = float(w_high.iloc[peak_pos])
        if peak_pos >= len(w_low)-1:
            continue
        after_peak_low   = w_low.iloc[peak_pos+1:]
        pullback_pos_rel = int(np.argmin(after_peak_low.values))
        pullback_low  = float(after_peak_low.iloc[pullback_pos_rel])

        days_since_pullback   = (w_close.index[-1] - after_peak_low.index[pullback_pos_rel]).days
        rebound_from_low_pct  = (latest_close / pullback_low - 1) * 100.0
        drop_a = (peak_high / pullback_low - 1) * 100.0
        drop_b = (peak_high - pullback_low) / peak_high * 100.0
        drop_pct = min(drop_a, drop_b)
        expected_rise_pct = (peak_high / latest_close - 1) * 100.0

        conds = [
            rebound_from_low_pct >= 5.0,
            drop_pct <= 15.0,
            days_since_pullback >= 2,
            latest_close >= latest_sma25,
            expected_rise_pct >= 3.0,
            latest_close >= pullback_low,
        ]
        if all(conds):
            results.append({
                "Ticker": ticker,
                "Expected_Upper": round(peak_high, 2),
                "Pullback_Low": round(pullback_low, 2),
                "Latest_Close": round(latest_close, 2),
                "Prev_Close": round(prev_close, 2),
                "Expected_Rise_%": round(expected_rise_pct, 2),
                "Return_%": round(expected_rise_pct, 2),
            })
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values(["Return_%"], ascending=[False]).reset_index(drop=True)

def run_pipeline():
    close, high, low = fetch_market_data(nikkei225_tickers, lookback_days=180)
    rs = []
    for w in (30, 14):
        df = find_pullback_candidates(close, high, low, window_days=w)
        if not df.empty:
            df["Window"] = w
            rs.append(df)
    if not rs:
        return pd.DataFrame()
    cat = pd.concat(rs, ignore_index=True).sort_values(["Ticker","Return_%"], ascending=[True, False])
    best = cat.groupby("Ticker", as_index=False).first().sort_values("Return_%", ascending=False).reset_index(drop=True)
    return best

# ===== 通知（4行レイアウト：変動率/底値比較は符号付き％）=====
def notify(best_df: pd.DataFrame, top_n=15):
    if best_df is None or best_df.empty:
        line_send("【押し目スクリーニング】本日は抽出なしでした。")
        return

    # フォーマッタ群
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

    # ヘッダ
    header = (
        f"📊【押し目スクリーニング】{datetime.now(TZ).strftime('%m/%d %H:%M')}\n"
        f"抽出: {len(best_df)} 銘柄（重複統合）\n"
        f"条件: 反発≥5%・下落≤15%・SMA25上・期待≥3%・2日経過\n"
        f"------------------------------\n"
    )
    send_long_text(header)

    cards = []
    for _, r in best_df.head(top_n).iterrows():
        ticker = r["Ticker"]; name = ticker_name_map.get(ticker, "")
        upper  = r.get("Expected_Upper")
        latest = r.get("Latest_Close")
        low    = r.get("Pullback_Low")
        rise_p = r.get("Expected_Rise_%")
        prev   = r.get("Prev_Close")

        # 期待額
        expect_amt = (float(upper) - float(latest)) if pd.notna(upper) and pd.notna(latest) else None
        # 変動率: (今/前日 - 1) * 100
        chg_pct = ((float(latest) / float(prev)) - 1) * 100 if (pd.notna(latest) and pd.notna(prev) and float(prev) != 0.0) else None
        # 底値比較: (今/底値 - 1) * 100
        bot_pct = ((float(latest) / float(low)) - 1) * 100 if (pd.notna(latest) and pd.notna(low) and float(low) != 0.0) else None

        line1 = f"{ticker} {name}".rstrip()
        line2 = f"↗ {fpct(rise_p)}   🎯 上 {fnum(upper)}   下 {fnum(low)}"
        line3 = f"今 {fnum(latest)}   🎯 期待額 {fnum(expect_amt)}"
        line4 = f"変動率 {fpct_signed(chg_pct)}   底値比較 {fpct_signed(bot_pct)}"

        cards.append("\n".join([line1, line2, line3, line4]))

    # 5銘柄ずつ送信
    for i in range(0, len(cards), 5):
        block = ("\n— — — — —\n").join(cards[i:i+5])
        send_long_text(block)

# ===== 取引日/時間判定（JST）=====
def is_trading_day_jst(dt: datetime):
    if dt.weekday() >= 5: return False
    if jpholiday.is_holiday(dt.date()): return False
    return True

def is_trading_time_jst(dt: datetime):
    h, m = dt.hour, dt.minute
    return (h > 9 or (h == 9 and m >= 0)) and (h < 15 or (h == 15 and m <= 30))

def main():
    now = datetime.now(TZ)
    force = os.getenv("FORCE_RUN") == "1"  # 手動実行のとき強制
    if not force:
        if not is_trading_day_jst(now) or not is_trading_time_jst(now):
            print(f"[SKIP] {now} 非取引時間")
            return
    best = run_pipeline()
    notify(best, top_n=15)

if __name__ == "__main__":
    main()
