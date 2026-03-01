
# =============================================
# Beta + Sector info helper for BBreturn notify
# Adds:
#   - S17 sector
#   - S33 sector
#   - Sector Beta
#   - Beta
#   - BetaScore
#
# Usage in your existing BBreturn.py:
#
#   from BBreturn_beta_with_sectors import beta_text
#
#   line = f"...既存の通知文..."
#   line += beta_text(ticker)
#
# =============================================

import os
import pandas as pd

BETA_DIR = os.path.join("reports", "beta")

STOCK_CSV = os.path.join(BETA_DIR, "beta_by_stock.csv")
SECTOR_CSV = os.path.join(BETA_DIR, "beta_by_sector.csv")

_stock_map = {}
_sector_map = {}

def _safe_float(x):
    try:
        return float(x)
    except:
        return None

def load_beta():
    global _stock_map, _sector_map

    if os.path.exists(STOCK_CSV):
        df = pd.read_csv(STOCK_CSV)
        for _, r in df.iterrows():
            code = str(r.get("Code", "")).zfill(4)
            _stock_map[code] = {
                "Beta": _safe_float(r.get("Beta")),
                "BetaScore": _safe_float(r.get("BetaScore")),
                "S17Nm": r.get("S17Nm", "-"),
                "S33Nm": r.get("S33Nm", "-"),
            }

    if os.path.exists(SECTOR_CSV):
        df = pd.read_csv(SECTOR_CSV)
        for _, r in df.iterrows():
            _sector_map[r.get("S17Nm")] = _safe_float(r.get("SectorBeta"))

# load once
load_beta()


def beta_text(ticker: str) -> str:
    """
    Return formatted beta + sector suffix for Discord message.
    """
    code = ticker.replace(".T", "").strip()

    info = _stock_map.get(code, {})

    s17 = info.get("S17Nm", "-")
    s33 = info.get("S33Nm", "-")

    beta = info.get("Beta")
    beta_score = info.get("BetaScore")
    sector_beta = _sector_map.get(s17)

    def fmt(v):
        return "-" if v is None else f"{v:.2f}"

    return (
        f" / S17:{s17}"
        f" / S33:{s33}"
        f" / Secβ:{fmt(sector_beta)}"
        f" / β:{fmt(beta)}"
        f" / βScore:{fmt(beta_score)}"
    )
