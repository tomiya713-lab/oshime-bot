# -*- coding: utf-8 -*-
"""
inv_hs_eval.py — 逆三尊の一括検出・勝率評価
要件対応:
- データ仕様: adjusted_close_prices_vertical.csv（縦持ち・日足）
- 除外: 決算当日+2営業日（同一ティッカー内の営業日基準）、2025年4月の検出
- 指標: ATR14, VolMA20
- 逆三尊検出: ピボット(谷/山)列、ヘッドの深さ(ATR正規化)、左右対称(値幅/時間)、ネックライン上抜け(+ε)かつ出来高急増
- 勝率(X=3,5,7): 検出日の終値を起点に10営業日以内の最高終値が +X% 到達 → 勝利
- 出力:
  1) サマリー（X=3,5,7の各集計）
  2) 2025年の月別検出件数（Xごと）
  3) 銘柄明細CSV（X=3,5,7 それぞれ）
- おまけ: しきい値緩和版（EPS_BREAKとVOL_SURGEを少し緩めたケース）も同じ要領で集計可能
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import List, Dict, Tuple

# ---------- 入力パス（必要なら変更） ----------
PRICE_PATH = "adjusted_close_prices_vertical.csv"
EARN_PATH  = "earnings_announcements_3y.csv"
SECTOR_PATH = "TickerList_with_JPX33_kabutan_tags.xlsx"

# ---------- パラメータ（アルゴリズム） ----------
W = 4               # ピボットの左右幅（2W+1窓の中心が極値）
K1 = 0.8            # ヘッドの深さ（min(L1,L3) - L2 >= K1*ATR）
K2 = 1.2            # 左右の“肩の深さ”対称性（ATR単位の許容差）
K3 = 0.35           # 左右の“時間間隔”対称性（全体期間に対する比率）
K4 = 1.0            # 肩の山の高さの近さ（ATR単位）
EPS_BREAK = 0.004   # ネックライン上抜けの余白（0.4%）
VOL_SURGE = 1.4     # ブレイク時の出来高が20MAの何倍以上か
LOOK_AHEAD = 30     # ブレイク確認の探索上限（H2から右へ最大30本）
SCORE_MIN = 60.0    # スコア閾値（簡易スコア）

# ---------- 勝率判定 ----------
WIN_THRESHOLDS = (3, 5, 7)    # %
WIN_WINDOW = 10               # 営業日以内

# ---------- 便利関数 ----------
def calc_atr14(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["Adj_Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()

def rolling_pivots(series_low: pd.Series, series_high: pd.Series, w: int) -> Tuple[pd.Series, pd.Series]:
    is_valley = (series_low.rolling(2*w+1, center=True)
                 .apply(lambda x: float(x[w] == x.min()), raw=True).fillna(0).astype(bool))
    is_peak   = (series_high.rolling(2*w+1, center=True)
                 .apply(lambda x: float(x[w] == x.max()), raw=True).fillna(0).astype(bool))
    return is_valley, is_peak

def line_value_at(t1: int, y1: float, t2: int, y2: float, t: int) -> float:
    if t2 == t1: return y1
    return y1 + (y2 - y1) * (t - t1) / (t2 - t1)

# ---------- 検出（単一ティッカー） ----------
def detect_inverse_hs_one(
    df_t: pd.DataFrame,
    w=W, k1=K1, k2=K2, k3=K3, k4=K4, eps_break=EPS_BREAK, vol_surge=VOL_SURGE,
    look_ahead=LOOK_AHEAD, score_min=SCORE_MIN
) -> List[Dict]:
    df = df_t.reset_index(drop=True).copy()
    if len(df) < (2*w+1)+25:
        return []

    df["ATR14"] = calc_atr14(df)
    df["VolMA20"] = df["volume"].rolling(20).mean()

    is_valley, is_peak = rolling_pivots(df["low"], df["high"], w=w)
    valleys = df[is_valley].index.tolist()
    peaks   = df[is_peak].index.tolist()

    out = []
    for i1 in range(len(valleys)-2):
        L1 = valleys[i1]
        H1_list = [p for p in peaks if p > L1]
        if not H1_list: continue
        H1 = H1_list[0]

        L2_list = [v for v in valleys if v > H1]
        if not L2_list: continue
        L2 = L2_list[0]

        H2_list = [p for p in peaks if p > L2]
        if not H2_list: continue
        H2 = H2_list[0]

        L3_list = [v for v in valleys if v > H2]
        if not L3_list: continue
        L3 = L3_list[0]

        atr_head = df.loc[L1:L3, "ATR14"].mean()
        if not np.isfinite(atr_head) or atr_head <= 0:
            continue

        L1_low, L2_low, L3_low = df.at[L1,"low"], df.at[L2,"low"], df.at[L3,"low"]
        H1_high, H2_high = df.at[H1,"high"], df.at[H2,"high"]

        # (a) ヘッドが最も深い（ATR基準）
        if not (L2_low <= min(L1_low, L3_low) - k1*atr_head):
            continue
        # (b) 左右“深さ”対称
        if abs((L1_low - L3_low)) > k2*atr_head:
            continue
        # (c) 左右“時間”対称
        if abs((L2-L1) - (L3-L2)) > k3*(L3 - L1 + 1):
            continue
        # (d) 肩の山の高さが近い
        if abs(H1_high - H2_high) > k4*atr_head:
            continue

        # ネックラインブレイク（H2の右側〜最大look_ahead本）
        broken = None
        for t in range(H2+1, min(L3+look_ahead, len(df))):
            neck_t = line_value_at(H1, H1_high, H2, H2_high, t)
            close_t = df.at[t,"Adj_Close"]
            vol_t   = df.at[t,"volume"]; volma = df.at[t,"VolMA20"]
            if (close_t >= neck_t * (1 + eps_break)) and (vol_t >= volma * vol_surge):
                broken = t; break
        if broken is None:
            continue

        # スコア（簡易）
        depth_sym = max(0, 1 - abs((L1_low-L3_low))/(k2*atr_head+1e-9))
        gap_sym   = max(0, 1 - abs((L2-L1)-(L3-L2))/(k3*(L3-L1+1e-9)))
        head_clr  = max(0, (min(L1_low,L3_low)-L2_low)/(k1*atr_head))
        neck_flat = max(0, 1 - abs(H1_high-H2_high)/(k4*atr_head+1e-9))
        left_vol  = df.loc[L1:H1, "volume"].mean()
        right_vol = df.loc[H2:L3, "volume"].mean() if L3>H2 else df.loc[H2:, "volume"].mean()
        vol_dry   = (right_vol/left_vol) if (left_vol>0 and np.isfinite(left_vol)) else 1.0
        vol_score = min(1.0, (df.at[broken,'volume']/(df.at[broken,'VolMA20']+1e-9))/2) * 0.6 \
                    + max(0, 1 - vol_dry) * 0.4

        S1 = 100*(0.5*depth_sym + 0.5*gap_sym)
        S2 = 100*(0.7*neck_flat + 0.3*(1 if H2_high>=H1_high else 0))
        S3 = 100*min(1.0, head_clr)
        S4 = 100*vol_score
        score = 0.25*S1 + 0.2*S2 + 0.25*S3 + 0.2*S4 + 0.1*0.5

        if score < score_min:
            continue

        out.append({
            "L1_idx": int(L1), "H1_idx": int(H1),
            "L2_idx": int(L2), "H2_idx": int(H2),
            "L3_idx": int(L3), "Break_idx": int(broken),
            "Date_L1": df.at[L1,"Date"], "Date_H1": df.at[H1,"Date"],
            "Date_L2": df.at[L2,"Date"], "Date_H2": df.at[H2,"Date"],
            "Date_L3": df.at[L3,"Date"], "Date_Break": df.at[broken,"Date"],
            "Detect_Close": float(df.at[broken,"Adj_Close"]),
            "Detect_Low": float(df.at[broken,"low"]),
            "Detect_High": float(df.at[broken,"high"]),
            "Score": round(float(score),1)
        })
    return out

# ---------- 主要処理 ----------
def main():
    # 1) 読み込み・前処理
    prices = pd.read_csv(PRICE_PATH, parse_dates=["Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)
    prices["Ticker"] = prices["Ticker"].astype(str)

    earn = pd.read_csv(EARN_PATH)
    if "提出日" in earn.columns:
        earn["Date"] = pd.to_datetime(earn["提出日"], errors="coerce")
    elif "date" in earn.columns:
        earn["Date"] = pd.to_datetime(earn["date"], errors="coerce")
    else:
        for c in earn.columns:
            if "date" in c.lower():
                earn["Date"] = pd.to_datetime(earn[c], errors="coerce")
                break
        if "Date" not in earn.columns:
            raise ValueError("earnings_announcements_3y.csv: 日付列（提出日 or date）が見つかりません。")
    earn["Ticker"] = earn.get("Ticker", earn.get("ticker")).astype(str)
    earn = earn.dropna(subset=["Date"])[["Ticker","Date"]].drop_duplicates()

    # 2) 決算除外リスト（当日+2営業日）
    prices["bidx"] = prices.groupby("Ticker").cumcount()
    earn = earn.merge(prices[["Ticker","Date","bidx"]], on=["Ticker","Date"], how="left").dropna(subset=["bidx"])
    earn["bidx"] = earn["bidx"].astype(int)
    excl = earn[["Ticker","bidx"]].copy()
    for k in [1,2]:
        tmp = earn.copy(); tmp["bidx"] = tmp["bidx"] + k
        excl = pd.concat([excl, tmp[["Ticker","bidx"]]], ignore_index=True)
    excl = excl.merge(prices[["Ticker","bidx","Date"]], on=["Ticker","bidx"], how="left")[["Ticker","Date"]].dropna().drop_duplicates()
    excl["ExcludeEarnings"] = True

    # 3) 逆三尊検出（ティッカー毎）
    rows = []
    for tic, g in prices.groupby("Ticker", sort=False):
        det = detect_inverse_hs_one(g)
        for d in det:
            d["Ticker"] = tic
            rows.append(d)
    sig = pd.DataFrame(rows)

    # 4) 除外（決算 / 2025-04）
    if not sig.empty:
        sig = sig.merge(excl, left_on=["Ticker","Date_Break"], right_on=["Ticker","Date"], how="left", indicator=True)
        sig = sig[sig["_merge"]=="left_only"].drop(columns=["_merge","Date","ExcludeEarnings"])
        sig = sig[~((sig["Date_Break"].dt.year==2025) & (sig["Date_Break"].dt.month==4))]

    # 5) 勝率評価 X = 3,5,7
    def eval_wins(sig_df: pd.DataFrame, X: int) -> Tuple[pd.DataFrame, Dict]:
        if sig_df.empty:
            return pd.DataFrame(), {"detected":0,"win_rate":None,"avg_win_gain":None}
        out = sig_df.copy()

        px = prices.sort_values(["Ticker","Date"]).copy()
        px["row_id"] = px.groupby("Ticker").cumcount()
        out = out.merge(px[["Ticker","Date","row_id","Adj_Close","high","low"]],
                        left_on=["Ticker","Date_Break"], right_on=["Ticker","Date"], how="left")
        out = out.rename(columns={
            "Adj_Close":"Detect_Close_chk","high":"Detect_High_chk","low":"Detect_Low_chk"
        }).drop(columns=["Date"])

        wins = []
        for r in out.itertuples():
            sub = px[(px["Ticker"]==r.Ticker) & (px["row_id"]>r.row_id) & (px["row_id"]<=r.row_id+WIN_WINDOW)]
            if sub.empty:
                wins.append((False, pd.NaT, np.nan, np.nan, np.nan, np.nan)); continue
            target = float(r.Detect_Close) * (1 + X/100.0)
            max_row = sub.loc[sub["Adj_Close"].idxmax()]
            max_close = float(max_row["Adj_Close"])
            win = bool(max_close >= target)
            wins.append((
                win, pd.Timestamp(max_row["Date"]),
                float(max_row["low"]), float(max_row["Adj_Close"]), float(max_row["high"]),
                (max_close/float(r.Detect_Close)-1.0) if win else np.nan
            ))
        out[["Win","Win_Date","Win_Low","Win_Close","Win_High","Win_Gain"]] = pd.DataFrame(wins, index=out.index)

        detected = len(out)
        win_rate = float(out["Win"].mean()*100.0) if detected>0 else None
        avg_win_gain = float(out.loc[out["Win"],"Win_Gain"].mean()*100.0) if (detected>0 and out["Win"].any()) else None

        # 出力列に整形
        out_export = out.rename(columns={
            "Ticker":"TickerID",
            "Date_Break":"検出日",
            "Detect_Low":"検出日の底値",
            "Detect_Close":"検出日の終値",
            "Detect_High":"検出日の高値",
            "Win_Date":"勝利日",
            "Win_Low":"勝利日の底値",
            "Win_Close":"勝利日の終値",
            "Win_High":"勝利日の高値",
            "Win_Gain":"勝利上昇率"
        })[[
            "TickerID","検出日","検出日の底値","検出日の終値","検出日の高値",
            "勝利日","勝利日の底値","勝利日の終値","勝利日の高値","勝利上昇率"
        ]]

        return out_export, {"detected":detected,"win_rate":win_rate,"avg_win_gain":avg_win_gain}

    summaries = {}
    details_paths = {}
    denom = prices["Date"].nunique() * prices["Ticker"].nunique()
    n_months = prices["Date"].dt.to_period("M").nunique()

    for X in WIN_THRESHOLDS:
        detail, stats = eval_wins(sig, X)
        # 発生比率・想定月次件数
        occ = (stats["detected"]/denom*100.0) if denom>0 else None
        per_month = stats["detected"]/n_months if n_months>0 else None
        # 月別 2025
        monthly = pd.Series(dtype=int)
        if not detail.empty:
            tmp = detail.copy()
            tmp["Year"] = pd.to_datetime(tmp["検出日"]).dt.year
            tmp["Month"] = pd.to_datetime(tmp["検出日"]).dt.month
            monthly = tmp[tmp["Year"]==2025].groupby("Month").size().reindex(range(1,13), fill_value=0)

        summaries[X] = {
            "合計検出件数": stats["detected"],
            "発生比率(%)": round(occ, 4) if occ is not None else None,
            "想定月次件数": round(per_month, 2) if per_month is not None else None,
            "勝率(%)": round(stats["win_rate"], 2) if stats["win_rate"] is not None else None,
            "勝利上昇率(勝者平均% )": round(stats["avg_win_gain"], 2) if stats["avg_win_gain"] is not None else None,
            "2025月別": monthly.to_dict()
        }

        # CSV出力
        csv_name = f"invHS_details_X{X}.csv"
        detail.to_csv(csv_name, index=False, encoding="utf-8-sig")
        details_paths[X] = csv_name

    # サマリー表示
    summary_table = pd.DataFrame.from_dict(summaries, orient="index")[
        ["合計検出件数","発生比率(%)","想定月次件数","勝率(%)","勝利上昇率(勝者平均% )","2025月別"]
    ].rename_axis("X(%)").reset_index()

    print("\n===== 逆三尊 サマリー =====")
    print(summary_table.to_string(index=False))

    print("\n===== CSV 出力 =====")
    for X, p in details_paths.items():
        print(f"X={X}%: {p}")

if __name__ == "__main__":
    main()
