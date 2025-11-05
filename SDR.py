# SDR.py
from __future__ import annotations
from typing import List, Optional, Dict
import time, io, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ================== COSTANTI ==================
FALLBACK_SCHEMA = "aedbdata"
ROW_LIMIT_PER_TABLE = 0  # 0 = tutte le righe
DARK_BG = "#0f1113"
FUXIA  = "#ff00ff"
GREEN  = "#00ff66"
ORANGE = "#ffa500"
CADUTI = "#4b5563"
BLUE   = "#3b82f6"
TEAL   = "#14b8a6"
YELLOW = "#f97316"
VIOLET = "#a855f7"
GREY_OVERLAY = "rgba(148,163,184,0.22)"

OP_STACK_COLORS = [
    "#ef4444", "#22c55e", "#3b82f6", "#eab308", "#a855f7",
    "#14b8a6", "#f97316", "#0ea5e9", "#f43f5e", "#6366f1",
    "#84cc16", "#facc15", "#ec4899", "#22d3ee", "#fb7185",
    "#0f766e", "#4b5563", "#fbbf24", "#1d4ed8", "#7c3aed"
]

# ========== LEGGENDE HTML (senza indentazione che crea code block) ==========
def _legend_linea_sett_html() -> str:
    return f"""<div style="display:flex;gap:56px;justify-content:center;align-items:center;margin-top:4px;margin-bottom:20px;">
<div style="display:flex;align-items:center;gap:10px;">
<span style="width:14px;height:14px;background:{FUXIA};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Ore conversazione</span>
</div>
<div style="display:flex;align-items:center;gap:10px;">
<span style="width:14px;height:14px;background:{GREEN};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Positivi</span>
</div>
</div>"""

def _legend_trend_sett_html() -> str:
    return f"""<div style="display:flex;gap:48px;justify-content:center;align-items:center;margin-top:4px;margin-bottom:20px;">
<div style="display:flex;align-items:center;gap:10px;">
<span style="width:14px;height:14px;background:{GREEN};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Trend positivi</span>
</div>
<div style="display:flex;align-items:center;gap:10px;">
<span style="width:14px;height:14px;background:{FUXIA};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Trend ore conversazione</span>
</div>
<div style="display:flex;align-items:center;gap:10px;">
<span style="width:14px;height:14px;background:{ORANGE};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Trend positivi confermati</span>
</div>
</div>"""

def _legend_trend_lav_html() -> str:
    return f"""<div style="display:flex;gap:56px;justify-content:center;align-items:center;margin-top:4px;margin-bottom:20px;">
<div style="display:flex;align-items:center;gap:10px;">
<span style="width:14px;height:14px;background:{GREEN};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Trend positivi</span>
</div>
<div style="display:flex;align-items:center;gap:10px;">
<span style="width:14px;height:14px;background:{FUXIA};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Trend ore lavorazione generale</span>
</div>
</div>"""

def _legend_bar_mese_html() -> str:
    return f"""<div style="display:flex;gap:42px;justify-content:center;align-items:center;margin-top:4px;margin-bottom:20px;">
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:{ORANGE};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Positivi confermati</span>
</div>
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:{CADUTI};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Caduti</span>
</div>
</div>"""

def _legend_lavorazioni_html() -> str:
    return f"""<div style="display:flex;gap:30px;justify-content:center;align-items:center;margin-top:4px;margin-bottom:20px;flex-wrap:wrap;">
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:{BLUE};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Lav. contatti</span>
</div>
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:{TEAL};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Lav. varie</span>
</div>
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:{YELLOW};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">In chiamata</span>
</div>
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:{FUXIA};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Conversazione</span>
</div>
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:{CADUTI};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">In attesa</span>
</div>
</div>"""

def _legend_bar_simple_html(has_trend: bool = False) -> str:
    trend_html = ""
    if has_trend:
        trend_html = f"""<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:{ORANGE};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Trend</span>
</div>"""
    return f"""<div style="display:flex;gap:28px;justify-content:center;align-items:center;margin-top:4px;margin-bottom:20px;flex-wrap:wrap;">
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:{GREEN};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Valore</span>
</div>
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:{VIOLET};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">Media</span>
</div>
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:18px;height:14px;background:{GREY_OVERLAY};display:inline-block;border-radius:3px;border:1px solid rgba(148,163,184,0.35);"></span>
<span style="font-size:13px;color:#ffffff;">Area ±5%</span>
</div>
{trend_html}
</div>"""

def _legend_bar_simple_html_blue(has_trend: bool = False) -> str:
    return _legend_bar_simple_html(has_trend).replace(f"background:{GREEN};", f"background:{BLUE};")

def _legend_bar_simple_html_orange(has_trend: bool = False) -> str:
    return _legend_bar_simple_html(has_trend).replace(f"background:{GREEN};", f"background:{ORANGE};")

def _legend_bar_simple_html_green(has_trend: bool = False) -> str:
    return _legend_bar_simple_html(has_trend)

def _legend_bar_simple_html_proc(has_trend: bool = False) -> str:
    return _legend_bar_simple_html(has_trend).replace(f"background:{GREEN};", "background:#22c55e;")

def _legend_red_mese_html() -> str:
    return """<div style="display:flex;gap:28px;justify-content:center;align-items:center;margin-top:4px;margin-bottom:20px;flex-wrap:wrap;">
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:#0ea5e9;display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">RED% (positivi)</span>
</div>
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:#f97316;display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">RED% (positivi confermati)</span>
</div>
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:38px;height:2px;background:#0ea5e9;display:inline-block;"></span>
<span style="font-size:13px;color:#ffffff;">Trend RED% (positivi)</span>
</div>
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:38px;height:2px;background:#f97316;display:inline-block;"></span>
<span style="font-size:13px;color:#ffffff;">Trend RED% (conf.)</span>
</div>
</div>"""

def _legend_red_operatore_html() -> str:
    return """<div style="display:flex;gap:28px;justify-content:center;align-items:center;margin-top:4px;margin-bottom:20px;flex-wrap:wrap;">
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:#0ea5e9;display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">RED% (positivi)</span>
</div>
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:#f97316;display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">RED% (positivi confermati)</span>
</div>
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:38px;height:2px;background:#0ea5e9;display:inline-block;"></span>
<span style="font-size:13px;color:#ffffff;">Media RED% (positivi)</span>
</div>
<div style="display:flex;align-items:center;gap:8px;">
<span style="width:38px;height:2px;background:#f97316;display:inline-block;"></span>
<span style="font-size:13px;color:#ffffff;">Media RED% (conf.)</span>
</div>
</div>"""

def _legend_dist_stack_html(ops: List[str], colors: List[str], label: str) -> str:
    items = []
    for op, col in zip(ops, colors):
        items.append(f"""<div style="display:flex;align-items:center;gap:8px;">
<span style="width:14px;height:14px;background:{col};display:inline-block;border-radius:3px;"></span>
<span style="font-size:13px;color:#ffffff;">{op}</span>
</div>""")
    return f"""<div style="display:flex;gap:18px;justify-content:center;align-items:center;margin-top:4px;margin-bottom:20px;flex-wrap:wrap;">
<div style="font-size:13px;color:#ffffff;margin-right:8px;">{label}</div>
{''.join(items)}
</div>"""

# alias campi principali
ALIAS_DATA = ["Data","data","date","Date","timestamp","Timestamp","datetime","dt","created_at","createdAt"]
ALIAS_ATT  = ["Attivita","attivita","Attività","attività","Activity","Categoria","TipoAttivita","Tipo","Task"]
ALIAS_OP   = ["Operatore","operatore","User","Agent","Utente","OperatoreNome","Operatore_Nome"]
ALIAS_CONV = [
    "conversazione","conversazioni","ore_conversazione","h_conversazione","durata_conversazione",
    "conversazioni_ore","ore","minuti_conversazione","durata_minuti","tempo_conversazione","talk_time"
]
ALIAS_POS  = ["positivi","positive","esiti_positivi","lead_positivi","ok","esito_positivo","positivi_tot"]
ALIAS_POS_CONF = [
    "Positivi confermati","positivi_confermati","confermati","ok_confermati","positivi_ok",
    "esito_confermato","positiviConfermati","positivi_confermate","confirmed","positivi_confirmed"
]
ALIAS_PROC = [
    "processati","processato","record_processati","contatti_processati",
    "elaborati","processed","calls","call_count","chiamate_processate"
]

ALIAS_LAV_GEN = [
    "lavorazione_generale","lavorazione generale","Lavorazione Generale",
    "tot_lavorazione","tot_lavorazioni","tempo_lavorazione","tempo_lavorazione_generale",
    "lavorazionegenerale","lav_generale","lav_generali","lavorazione_totale"
]
ALIAS_LAV_CONTATTI = [
    "lavorazione_contatti","lavorazione contatti","Lavorazione Contatti",
    "lav_contatti","tempo_contatti","contatti_lavorati"
]
ALIAS_LAV_VARIE = [
    "lavorazione_varie","lavorazione varie","Lavorazione Varie",
    "lav_varie","tempo_varie","varie_lavorate"
]
ALIAS_IN_CHIAMATA = [
    "in_chiamata","In chiamata","in chiamata","tempo_in_chiamata","tempo_chiamata","durata_chiamata"
]
ALIAS_IN_ATTESA = [
    "in_attesa_di_chiamata","in attesa di chiamta","In attesa di chiamata",
    "attesa_chiamata","tempo_attesa","in_attesa","attesa"
]

# ================== DB-FREE / CSV / EMBEDDED LOADER HELPERS ==================
PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"

def _try_get_injected_table_names():
    try:
        gl_tables = globals().get("tables", None)
        if isinstance(gl_tables, dict):
            return list(gl_tables.keys())
    except Exception:
        pass
    return []

def _try_get_injected_table(name: str) -> Optional[pd.DataFrame]:
    try:
        gf = globals().get("get_table", None)
        if callable(gf):
            df = gf(name)
            if isinstance(df, pd.DataFrame):
                return df
    except Exception:
        pass
    try:
        gfn = globals().get("get_table_norm", None)
        if callable(gfn):
            df = gfn(name)
            if isinstance(df, pd.DataFrame):
                return df
    except Exception:
        pass
    try:
        gl_tables = globals().get("tables", None)
        if isinstance(gl_tables, dict) and name in gl_tables:
            df = gl_tables[name]
            if isinstance(df, pd.DataFrame):
                return df
    except Exception:
        pass
    return None

def _try_dataset_embedded_names():
    try:
        import dataset_embedded as de  # type: ignore
        return de.list_tables()
    except Exception:
        return []

def _try_dataset_embedded_load(name: str) -> Optional[pd.DataFrame]:
    try:
        import dataset_embedded as de  # type: ignore
        if name in de.list_tables():
            return de.load_table(name)
    except Exception:
        return None

def _try_csv_load(name: str) -> Optional[pd.DataFrame]:
    p1 = DATA_DIR / f"{name}.csv"
    p2 = Path.cwd() / "data" / f"{name}.csv"
    candidates = [p1, p2]
    for p in candidates:
        try:
            if p.exists():
                for sep in [',', ';', '\t', '|']:
                    try:
                        df = pd.read_csv(p, sep=sep, engine="python")
                        if df.shape[1] > 1 or df.shape[0] > 0:
                            return df
                    except Exception:
                        continue
        except Exception:
            continue
    return None

# ================== DB-FREE REPLACEMENTS ==================
@st.cache_data(ttl=60, show_spinner=False)
def _current_db() -> str:
    return FALLBACK_SCHEMA

@st.cache_data(ttl=60, show_spinner=False)
def _db_version(schema: str) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

@st.cache_data(ttl=60, show_spinner=False)
def _list_tables(schema: str, vkey: str) -> pd.DataFrame:
    rows = []
    try:
        gl_tables = globals().get("tables", None)
        if isinstance(gl_tables, dict) and gl_tables:
            for name, df in gl_tables.items():
                try:
                    nrows = int(df.shape[0]) if hasattr(df, "shape") else 0
                except Exception:
                    nrows = 0
                rows.append({"table_name": name, "table_type": "BASE TABLE", "table_rows": nrows})
            return pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
    except Exception:
        pass
    try:
        names = _try_dataset_embedded_names()
        for n in names:
            rows.append({"table_name": n, "table_type": "BASE TABLE", "table_rows": 0})
        if rows:
            return pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
    except Exception:
        pass
    try:
        if DATA_DIR.exists():
            for p in sorted(DATA_DIR.glob("*.csv")):
                rows.append({"table_name": p.stem, "table_type": "CSV", "table_rows": 0})
            if rows:
                return pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame(columns=["table_name","table_type","table_rows"])

@st.cache_data(ttl=60, show_spinner=False)
def _load_table(schema: str, table: str, limit: int, vkey: str) -> pd.DataFrame:
    # 1) injected
    try:
        df = _try_get_injected_table(table)
        if isinstance(df, pd.DataFrame):
            return df.head(limit) if limit and limit > 0 else df
    except Exception:
        pass
    # 2) dataset_embedded
    try:
        df = _try_dataset_embedded_load(table)
        if isinstance(df, pd.DataFrame):
            return df.head(limit) if limit and limit > 0 else df
    except Exception:
        pass
    # 3) local CSV
    try:
        df = _try_csv_load(table)
        if isinstance(df, pd.DataFrame):
            return df.head(limit) if limit and limit > 0 else df
    except Exception:
        pass
    return pd.DataFrame()

# ================== COERCIZIONI / MATCH ==================
def _find_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    s_cols = set(cols)
    for a in aliases:
        if a in s_cols:
            return a
    low_map = {c.lower(): c for c in cols}
    for a in aliases:
        la = a.lower()
        if la in low_map:
            return low_map[la]
    for a in aliases:
        la = a.lower()
        for c in cols:
            if la in c.lower():
                return c
    return None

def _find_col_contains(df: pd.DataFrame, sub: str) -> Optional[str]:
    sub = sub.lower()
    for c in df.columns:
        if sub in c.lower():
            return c
    return None

def _to_datetime_fast(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s): return s
    if s.dtype == object:
        sv = s.replace({"0000-00-00": None, "0000-00-00 00:00:00": None, "": None})
        return pd.to_datetime(sv, errors="coerce", utc=False)
    return pd.to_datetime(s, errors="coerce", utc=False)

def _to_numeric_fast(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s): return s
    return pd.to_numeric(
        s.astype(str).str.replace(".", "", regex=False)
                     .str.replace(",", ".", regex=False)
                     .str.replace(" ", "", regex=False),
        errors="coerce",
    )

def _to_hours(s: pd.Series) -> pd.Series:
    if s.dtype == object and s.astype(str).str.contains(":").any():
        td = pd.to_timedelta(s.astype(str), errors="coerce")
        return td.dt.total_seconds() / 3600.0
    num = _to_numeric_fast(s)
    vals = num.dropna()
    if not vals.empty and (vals.mod(1) == 0).mean() > 0.95 and (vals.median() > 59):
        return num / 60.0
    return num

def show_df(df: pd.DataFrame, height: int | None = None):
    _df = df if df is not None else pd.DataFrame()
    n = len(_df)
    h = min(60 + 28 * max(1, min(n, 250)), 700)
    st.dataframe(_df, use_container_width=True, height=height or h)

def go_home():
    st.session_state["page"] = "home"

def _short_name(full: str) -> str:
    if not full:
        return ""
    parts = str(full).strip().split()
    if len(parts) == 1:
        return parts[0].title()
    last = " ".join(parts[:-1]).title()
    first = parts[-1]
    initial = first[0].upper() if first else ""
    return f"{last} {initial}."

# ================== FILTRI GLOBALI ==================
@st.cache_data(ttl=60, show_spinner=False)
def _collect_filter_options(schema_now: str, vkey: str):
    mesi, att, op = set(), set(), set()
    tbls = _list_tables(schema_now, vkey)
    for _, r in tbls.iterrows():
        df = _load_table(schema_now, r["table_name"], ROW_LIMIT_PER_TABLE, vkey)
        if df.empty:
            continue
        dcol = _find_col(df, ALIAS_DATA)
        if dcol:
            dt = _to_datetime_fast(df[dcol])
            if dt.notna().any():
                mesi.update(dt.dt.to_period("M").astype(str).dropna().unique().tolist())
        acol = _find_col(df, ALIAS_ATT)
        if acol:
            att.update(df[acol].dropna().astype(str).unique().tolist())
        ocol = _find_col(df, ALIAS_OP)
        if ocol:
            op.update(df[ocol].dropna().astype(str).unique().tolist())
    return sorted(mesi), sorted(att), sorted(op)

# ================== WEEKLY (CONV/POS) ==================
@st.cache_data(ttl=60, show_spinner=False)
def _weekly_conv_positivi_filtered(schema_now: str, vkey: str,
                                   mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    rows = []
    tbls = _list_tables(schema_now, vkey)
    for _, r in tbls.iterrows():
        df = _load_table(schema_now, r["table_name"], ROW_LIMIT_PER_TABLE, vkey)
        if df.empty:
            continue
        dcol = _find_col(df, ALIAS_DATA)
        if not dcol:
            continue
        acol = _find_col(df, ALIAS_ATT)
        ocol = _find_col(df, ALIAS_OP)
        ccol = _find_col(df, ALIAS_CONV)
        pcol = _find_col(df, ALIAS_POS)
        if not ccol and not pcol:
            continue
        if att_sel and not acol:
            continue
        if op_sel and not ocol:
            continue
        mask = pd.Series(True, index=df.index)
        dt = _to_datetime_fast(df[dcol])
        if mesi_sel:
            months = dt.dt.to_period("M").astype(str)
            mask &= months.isin(mesi_sel)
        if att_sel and acol:
            mask &= df[acol].astype(str).isin(att_sel)
        if op_sel and ocol:
            mask &= df[ocol].astype(str).isin(op_sel)
        if not mask.any():
            continue
        week = dt[mask].dt.to_period("W-MON").dt.start_time.dt.date
        conv = _to_hours(df.loc[mask, ccol]) if ccol else pd.Series(0, index=df.index[mask], dtype=float)
        pos = _to_numeric_fast(df.loc[mask, pcol]) if pcol else pd.Series(0, index=df.index[mask], dtype=float)
        tmp = pd.DataFrame({"week_start": week, "Ore conversazione": conv, "Positivi": pos})
        tmp = tmp.dropna(subset=["week_start"])
        if not tmp.empty:
            rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["week_start", "Ore conversazione", "Positivi"])
    all_df = pd.concat(rows, ignore_index=True)
    agg = (all_df.groupby("week_start", as_index=False)
           .agg({"Ore conversazione": "sum", "Positivi": "sum"})
           .sort_values("week_start"))
    return agg

# ================== WEEKLY FOR TREND ==================
@st.cache_data(ttl=60, show_spinner=False)
def _weekly_for_trend(schema_now: str, vkey: str,
                      mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    rows = []
    tbls = _list_tables(schema_now, vkey)
    for _, r in tbls.iterrows():
        df = _load_table(schema_now, r["table_name"], ROW_LIMIT_PER_TABLE, vkey)
        if df.empty:
            continue
        dcol = _find_col(df, ALIAS_DATA)
        if not dcol:
            continue
        acol = _find_col(df, ALIAS_ATT)
        ocol = _find_col(df, ALIAS_OP)
        ccol = _find_col(df, ALIAS_CONV)
        pcol = _find_col(df, ALIAS_POS)
        cfp  = _find_col(df, ALIAS_POS_CONF)
        if not any([ccol, pcol, cfp]):
            continue
        if att_sel and not acol:
            continue
        if op_sel and not ocol:
            continue
        mask = pd.Series(True, index=df.index)
        dt = _to_datetime_fast(df[dcol])
        if mesi_sel:
            months = dt.dt.to_period("M").astype(str)
            mask &= months.isin(mesi_sel)
        if att_sel and acol:
            mask &= df[acol].astype(str).isin(att_sel)
        if op_sel and ocol:
            mask &= df[ocol].astype(str).isin(op_sel)
        if not mask.any():
            continue
        week = dt[mask].dt.to_period("W-MON").dt.start_time.dt.date
        conv = _to_hours(df.loc[mask, ccol]) if ccol else pd.Series(0, index=df.index[mask], dtype=float)
        pos = _to_numeric_fast(df.loc[mask, pcol]) if pcol else pd.Series(0, index=df.index[mask], dtype=float)
        posc = _to_numeric_fast(df.loc[mask, cfp]) if cfp else pd.Series(0, index=df.index[mask], dtype=float)
        tmp = pd.DataFrame({"week_start": week, "conv": conv, "pos": pos, "pos_conf": posc})
        tmp = tmp.dropna(subset=["week_start"])
        if not tmp.empty:
            rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["week_start", "conv", "pos", "pos_conf"])
    all_df = pd.concat(rows, ignore_index=True)
    agg = (all_df.groupby("week_start", as_index=False)
           .agg({"conv": "sum", "pos": "sum", "pos_conf": "sum"})
           .sort_values("week_start"))
    return agg

# ================== WEEKLY FOR TREND LAV ==================
@st.cache_data(ttl=60, show_spinner=False)
def _weekly_for_trend_lav(schema_now: str, vkey: str,
                          mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    rows = []
    tbls = _list_tables(schema_now, vkey)
    for _, r in tbls.iterrows():
        df = _load_table(schema_now, r["table_name"], ROW_LIMIT_PER_TABLE, vkey)
        if df.empty:
            continue
        dcol = _find_col(df, ALIAS_DATA)
        lavg = _find_col(df, ALIAS_LAV_GEN) or _find_col_contains(df, "lavorazione")
        pcol = _find_col(df, ALIAS_POS)
        if not dcol or (not lavg and not pcol):
            continue
        acol = _find_col(df, ALIAS_ATT)
        ocol = _find_col(df, ALIAS_OP)
        if att_sel and not acol:
            continue
        if op_sel and not ocol:
            continue
        mask = pd.Series(True, index=df.index)
        dt = _to_datetime_fast(df[dcol])
        if mesi_sel:
            months = dt.dt.to_period("M").astype(str)
            mask &= months.isin(mesi_sel)
        if att_sel and acol:
            mask &= df[acol].astype(str).isin(att_sel)
        if op_sel and ocol:
            mask &= df[ocol].astype(str).isin(op_sel)
        if not mask.any():
            continue
        week = dt[mask].dt.to_period("W-MON").dt.start_time.dt.date
        lav = _to_hours(df.loc[mask, lavg]) if lavg else pd.Series(0, index=df.index[mask], dtype=float)
        pos = _to_numeric_fast(df.loc[mask, pcol]) if pcol else pd.Series(0, index=df.index[mask], dtype=float)
        tmp = pd.DataFrame({"week_start": week, "lav_gen": lav, "pos": pos}).dropna(subset=["week_start"])
        if not tmp.empty:
            rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["week_start","lav_gen","pos"])
    all_df = pd.concat(rows, ignore_index=True)
    agg = (all_df.groupby("week_start", as_index=False)
           .agg({"lav_gen":"sum","pos":"sum"})
           .sort_values("week_start"))
    return agg

# ================== PER MESE / PER OPERATORE (POSITIVI) ==================
@st.cache_data(ttl=60, show_spinner=False)
def _monthly_pos_conf(schema_now: str, vkey: str,
                      mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    rows = []
    tbls = _list_tables(schema_now, vkey)
    for _, r in tbls.iterrows():
        df = _load_table(schema_now, r["table_name"], ROW_LIMIT_PER_TABLE, vkey)
        if df.empty:
            continue
        dcol = _find_col(df, ALIAS_DATA)
        if not dcol:
            continue
        acol = _find_col(df, ALIAS_ATT)
        ocol = _find_col(df, ALIAS_OP)
        pcol = _find_col(df, ALIAS_POS)
        cfp  = _find_col(df, ALIAS_POS_CONF)
        if not pcol and not cfp:
            continue
        mask = pd.Series(True, index=df.index)
        dt = _to_datetime_fast(df[dcol])
        month = dt.dt.to_period("M").astype(str)
        if mesi_sel:
            mask &= month.isin(mesi_sel)
        if att_sel and acol:
            mask &= df[acol].astype(str).isin(att_sel)
        if op_sel and ocol:
            mask &= df[ocol].astype(str).isin(op_sel)
        if not mask.any():
            continue
        pos = _to_numeric_fast(df.loc[mask, pcol]) if pcol else pd.Series(0, index=df.index[mask], dtype=float)
        posc = _to_numeric_fast(df.loc[mask, cfp]) if cfp else pd.Series(0, index=df.index[mask], dtype=float)
        tmp = pd.DataFrame({
            "mese": month[mask],
            "positivi": pos,
            "positivi_conf": posc
        }).dropna(subset=["mese"])
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["mese","positivi","positivi_conf","caduti","conf_pct","caduti_pct"]).sort_values("mese")
    all_df = pd.concat(rows, ignore_index=True)
    agg = (all_df.groupby("mese", as_index=False)
           .agg({"positivi":"sum","positivi_conf":"sum"}))
    agg["caduti"] = (agg["positivi"] - agg["positivi_conf"]).clip(lower=0)
    agg["conf_pct"] = np.where(agg["positivi"]>0, agg["positivi_conf"]/agg["positivi"]*100, 0)
    agg["caduti_pct"] = 100 - agg["conf_pct"]
    return agg.sort_values("mese")

@st.cache_data(ttl=60, show_spinner=False)
def _operator_pos_conf(schema_now: str, vkey: str,
                       mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    rows = []
    tbls = _list_tables(schema_now, vkey)
    for _, r in tbls.iterrows():
        df = _load_table(schema_now, r["table_name"], ROW_LIMIT_PER_TABLE, vkey)
        if df.empty:
            continue
        dcol = _find_col(df, ALIAS_DATA)
        ocol = _find_col(df, ALIAS_OP)
        acol = _find_col(df, ALIAS_ATT)
        pcol = _find_col(df, ALIAS_POS)
        cfp  = _find_col(df, ALIAS_POS_CONF)
        if not ocol:
            continue
        if not pcol and not cfp:
            continue
        mask = pd.Series(True, index=df.index)
        if dcol:
            dt = _to_datetime_fast(df[dcol])
            month = dt.dt.to_period("M").astype(str)
            if mesi_sel:
                mask &= month.isin(mesi_sel)
        if att_sel and acol:
            mask &= df[acol].astype(str).isin(att_sel)
        if op_sel:
            mask &= df[ocol].astype(str).isin(op_sel)
        if not mask.any():
            continue
        pos = _to_numeric_fast(df.loc[mask, pcol]) if pcol else pd.Series(0, index=df.index[mask], dtype=float)
        posc = _to_numeric_fast(df.loc[mask, cfp]) if cfp else pd.Series(0, index=df.index[mask], dtype=float)
        tmp = pd.DataFrame({
            "operatore": df.loc[mask, ocol].astype(str),
            "operatore_short": df.loc[mask, ocol].astype(str).map(_short_name),
            "positivi": pos,
            "positivi_conf": posc
        })
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=[
            "operatore","operatore_short","positivi","positivi_conf","caduti","conf_pct","caduti_pct"
        ])
    all_df = pd.concat(rows, ignore_index=True)
    agg = (all_df.groupby(["operatore","operatore_short"], as_index=False)
           .agg({"positivi":"sum","positivi_conf":"sum"}))
    agg["caduti"] = (agg["positivi"] - agg["positivi_conf"]).clip(lower=0)
    agg["conf_pct"] = np.where(agg["positivi"]>0, agg["positivi_conf"]/agg["positivi"]*100, 0)
    agg["caduti_pct"] = 100 - agg["conf_pct"]
    return agg.sort_values("operatore")

# ================== LAVORAZIONI ==================
@st.cache_data(ttl=60, show_spinner=False)
def _monthly_lavorazioni(schema_now: str, vkey: str,
                         mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    rows = []
    tbls = _list_tables(schema_now, vkey)
    for _, r in tbls.iterrows():
        df = _load_table(schema_now, r["table_name"], ROW_LIMIT_PER_TABLE, vkey)
        if df.empty:
            continue
        dcol = _find_col(df, ALIAS_DATA)
        if not dcol:
            continue
        lavg = _find_col(df, ALIAS_LAV_GEN) or _find_col_contains(df, "lavorazione")
        if not lavg:
            continue
        acol = _find_col(df, ALIAS_ATT)
        ocol = _find_col(df, ALIAS_OP)
        lavc = _find_col(df, ALIAS_LAV_CONTATTI) or _find_col_contains(df, "contatti")
        lavv = _find_col(df, ALIAS_LAV_VARIE) or _find_col_contains(df, "varie")
        inch = _find_col(df, ALIAS_IN_CHIAMATA) or _find_col_contains(df, "chiamata")
        inatt = _find_col(df, ALIAS_IN_ATTESA) or _find_col_contains(df, "attesa")
        conv = _find_col(df, ALIAS_CONV)
        mask = pd.Series(True, index=df.index)
        dt = _to_datetime_fast(df[dcol])
        month = dt.dt.to_period("M").astype(str)
        if mesi_sel:
            mask &= month.isin(mesi_sel)
        if att_sel and acol:
            mask &= df[acol].astype(str).isin(att_sel)
        if op_sel and ocol:
            mask &= df[ocol].astype(str).isin(op_sel)
        if not mask.any():
            continue
        base = _to_hours(df.loc[mask, lavg])
        c1 = _to_hours(df.loc[mask, lavc]) if lavc else pd.Series(0, index=base.index)
        c2 = _to_hours(df.loc[mask, lavv]) if lavv else pd.Series(0, index=base.index)
        c3 = _to_hours(df.loc[mask, inch]) if inch else pd.Series(0, index=base.index)
        c4 = _to_hours(df.loc[mask, conv]) if conv else pd.Series(0, index=base.index)
        c5 = _to_hours(df.loc[mask, inatt]) if inatt else pd.Series(0, index=base.index)
        tmp = pd.DataFrame({
            "mese": month[mask],
            "lav_gen": base,
            "lav_contatti": c1,
            "lav_varie": c2,
            "in_chiamata": c3,
            "conversazione": c4,
            "in_attesa": c5,
        }).dropna(subset=["mese"])
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=[
            "mese","lav_gen","lav_contatti","lav_varie","in_chiamata","conversazione","in_attesa",
            "pct_contatti","pct_varie","pct_in_ch","pct_conv","pct_attesa"
        ])
    all_df = pd.concat(rows, ignore_index=True)
    agg = (all_df.groupby("mese", as_index=False)
           .sum(numeric_only=True))
    tot = agg["lav_gen"].replace(0, np.nan)
    agg["pct_contatti"] = np.where(tot.notna(), agg["lav_contatti"]/tot*100, 0)
    agg["pct_varie"] = np.where(tot.notna(), agg["lav_varie"]/tot*100, 0)
    agg["pct_in_ch"] = np.where(tot.notna(), agg["in_chiamata"]/tot*100, 0)
    agg["pct_conv"] = np.where(tot.notna(), agg["conversazione"]/tot*100, 0)
    agg["pct_attesa"] = np.where(tot.notna(), agg["in_attesa"]/tot*100, 0)
    agg = agg.fillna(0).sort_values("mese")
    return agg

@st.cache_data(ttl=60, show_spinner=False)
def _operator_lavorazioni(schema_now: str, vkey: str,
                          mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    rows = []
    tbls = _list_tables(schema_now, vkey)
    for _, r in tbls.iterrows():
        df = _load_table(schema_now, r["table_name"], ROW_LIMIT_PER_TABLE, vkey)
        if df.empty:
            continue
        ocol = _find_col(df, ALIAS_OP)
        if not ocol:
            continue
        dcol = _find_col(df, ALIAS_DATA)
        lavg = _find_col(df, ALIAS_LAV_GEN) or _find_col_contains(df, "lavorazione")
        if not lavg:
            continue
        acol = _find_col(df, ALIAS_ATT)
        lavc = _find_col(df, ALIAS_LAV_CONTATTI) or _find_col_contains(df, "contatti")
        lavv = _find_col(df, ALIAS_LAV_VARIE) or _find_col_contains(df, "varie")
        inch = _find_col(df, ALIAS_IN_CHIAMATA) or _find_col_contains(df, "chiamata")
        inatt = _find_col(df, ALIAS_IN_ATTESA) or _find_col_contains(df, "attesa")
        conv = _find_col(df, ALIAS_CONV)
        mask = pd.Series(True, index=df.index)
        if dcol:
            dt = _to_datetime_fast(df[dcol])
            month = dt.dt.to_period("M").astype(str)
            if mesi_sel:
                mask &= month.isin(mesi_sel)
        if att_sel and acol:
            mask &= df[acol].astype(str).isin(att_sel)
        if op_sel:
            mask &= df[ocol].astype(str).isin(op_sel)
        if not mask.any():
            continue
        base = _to_hours(df.loc[mask, lavg])
        c1 = _to_hours(df.loc[mask, lavc]) if lavc else pd.Series(0, index=base.index)
        c2 = _to_hours(df.loc[mask, lavv]) if lavv else pd.Series(0, index=base.index)
        c3 = _to_hours(df.loc[mask, inch]) if inch else pd.Series(0, index=base.index)
        c4 = _to_hours(df.loc[mask, conv]) if conv else pd.Series(0, index=base.index)
        c5 = _to_hours(df.loc[mask, inatt]) if inatt else pd.Series(0, index=base.index)
        tmp = pd.DataFrame({
            "operatore": df.loc[mask, ocol].astype(str),
            "operatore_short": df.loc[mask, ocol].astype(str).map(_short_name),
            "lav_gen": base,
            "lav_contatti": c1,
            "lav_varie": c2,
            "in_chiamata": c3,
            "conversazione": c4,
            "in_attesa": c5,
        })
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=[
            "operatore","operatore_short","lav_gen","lav_contatti","lav_varie","in_chiamata","conversazione","in_attesa",
            "pct_contatti","pct_varie","pct_in_ch","pct_conv","pct_attesa"
        ])
    all_df = pd.concat(rows, ignore_index=True)
    agg = (all_df.groupby(["operatore","operatore_short"], as_index=False)
           .sum(numeric_only=True))
    tot = agg["lav_gen"].replace(0, np.nan)
    agg["pct_contatti"] = np.where(tot.notna(), agg["lav_contatti"]/tot*100, 0)
    agg["pct_varie"] = np.where(tot.notna(), agg["lav_varie"]/tot*100, 0)
    agg["pct_in_ch"] = np.where(tot.notna(), agg["in_chiamata"]/tot*100, 0)
    agg["pct_conv"] = np.where(tot.notna(), agg["conversazione"]/tot*100, 0)
    agg["pct_attesa"] = np.where(tot.notna(), agg["in_attesa"]/tot*100, 0)
    agg = agg.fillna(0).sort_values("operatore")
    return agg

# ====== NUOVO: LAVORAZIONE PER MESE E OPERATORE ======
@st.cache_data(ttl=60, show_spinner=False)
def _monthly_operator_lavorazioni(schema_now: str, vkey: str,
                                  mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    rows = []
    tbls = _list_tables(schema_now, vkey)
    for _, r in tbls.iterrows():
        df = _load_table(schema_now, r["table_name"], ROW_LIMIT_PER_TABLE, vkey)
        if df.empty:
            continue
        ocol = _find_col(df, ALIAS_OP)
        dcol = _find_col(df, ALIAS_DATA)
        lavg = _find_col(df, ALIAS_LAV_GEN) or _find_col_contains(df, "lavorazione")
        inch = _find_col(df, ALIAS_IN_CHIAMATA) or _find_col_contains(df, "chiamata")
        if not (ocol and dcol and (lavg or inch)):
            continue
        acol = _find_col(df, ALIAS_ATT)
        dt = _to_datetime_fast(df[dcol])
        month = dt.dt.to_period("M").astype(str)
        mask = pd.Series(True, index=df.index)
        if mesi_sel:
            mask &= month.isin(mesi_sel)
        if att_sel and acol:
            mask &= df[acol].astype(str).isin(att_sel)
        if op_sel:
            mask &= df[ocol].astype(str).isin(op_sel)
        if not mask.any():
            continue
        base = _to_hours(df.loc[mask, lavg]) if lavg else pd.Series(0, index=df.index[mask], dtype=float)
        inch_s = _to_hours(df.loc[mask, inch]) if inch else pd.Series(0, index=df.index[mask], dtype=float)
        tmp = pd.DataFrame({
            "mese": month[mask],
            "operatore": df.loc[mask, ocol].astype(str),
            "operatore_short": df.loc[mask, ocol].astype(str).map(_short_name),
            "lav_gen": base,
            "in_chiamata": inch_s,
        }).dropna(subset=["mese"])
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["mese","operatore","operatore_short","lav_gen","in_chiamata"])
    all_df = pd.concat(rows, ignore_index=True)
    agg = (all_df.groupby(["mese","operatore","operatore_short"], as_index=False)
           .sum(numeric_only=True))
    return agg.sort_values(["mese","operatore"])

# ================== IN CHIAMATA / LAV. GENERALE (FUNZIONI) ==================
@st.cache_data(ttl=60, show_spinner=False)
def _inch_lav_mese(schema_now: str, vkey: str,
                   mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    df_lav = _monthly_lavorazioni(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_lav.empty:
        return pd.DataFrame(columns=["mese","ratio_pct"])
    df = df_lav[["mese","in_chiamata","lav_gen"]].copy()
    df["ratio_pct"] = np.where(df["lav_gen"]>0, df["in_chiamata"]/df["lav_gen"]*100, 0)
    return df.sort_values("mese")

@st.cache_data(ttl=60, show_spinner=False)
def _inch_lav_operatore(schema_now: str, vkey: str,
                        mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    df_lav = _operator_lavorazioni(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_lav.empty:
        return pd.DataFrame(columns=["operatore_short","ratio_pct"])
    df = df_lav[["operatore","operatore_short","in_chiamata","lav_gen"]].copy()
    df["ratio_pct"] = np.where(df["lav_gen"]>0, df["in_chiamata"]/df["lav_gen"]*100, 0)
    return df.sort_values("operatore")

# ================== GRAFICI e RESTO ==================
def _weekly_chart(dfw: pd.DataFrame) -> go.Figure:
    dfp = dfw.copy()
    while len(dfp) > 0 and dfp["Ore conversazione"].iloc[-1] == 0 and dfp["Positivi"].iloc[-1] == 0:
        dfp = dfp.iloc[:-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfp["week_start"], y=dfp["Ore conversazione"],
        mode="lines+markers",
        line=dict(width=2, color=FUXIA),
        marker=dict(size=6, symbol="square", color=FUXIA),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=dfp["week_start"], y=dfp["Positivi"],
        mode="lines+markers",
        line=dict(width=2, color=GREEN),
        marker=dict(size=6, symbol="circle", color=GREEN),
        showlegend=False,
    ))
    fig.update_layout(
        title=dict(text="Andamento settimanale: Ore conversazione vs Positivi", pad=dict(t=6)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=520,
        xaxis_title="", yaxis_title="Valore",
        showlegend=False,
        margin=dict(t=70, l=60, r=60, b=120),
    )
    return fig

def _trend_chart_lines(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    def _reg(df, col):
        if df.empty or col not in df.columns: return None
        y = df[col].astype(float).fillna(0.0).to_numpy()
        if np.allclose(y, 0): return None
        x_dates = pd.to_datetime(df["week_start"])
        x_num = x_dates.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
        if len(x_num) < 2: return None
        idx = np.argsort(x_num)
        x_num = x_num[idx]
        y = y[idx]
        a, b = np.polyfit(x_num, y, 1)
        x0, x1 = x_num.min(), x_num.max()
        y0, y1 = a * x0 + b, a * x1 + b
        return pd.DataFrame({
            "x": [pd.Timestamp.fromordinal(int(x0)), pd.Timestamp.fromordinal(int(x1))],
            "y": [y0, y1],
        })
    rc = _reg(df, "conv")
    if rc is not None:
        fig.add_trace(go.Scatter(
            x=rc["x"], y=rc["y"],
            mode="lines",
            line=dict(width=3, color=FUXIA),
            showlegend=False,
        ))
    rp = _reg(df, "pos")
    if rp is not None:
        fig.add_trace(go.Scatter(
            x=rp["x"], y=rp["y"],
            mode="lines",
            line=dict(width=3, color=GREEN),
            showlegend=False,
        ))
    rconf = _reg(df, "pos_conf")
    if rconf is not None:
        fig.add_trace(go.Scatter(
            x=rconf["x"], y=rconf["y"],
            mode="lines",
            line=dict(width=3, color=ORANGE),
            showlegend=False,
        ))
    fig.update_layout(
        title=dict(text="Trend settimanale (rette di regressione)", pad=dict(t=6)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=520,
        xaxis_title="", yaxis_title="Valore",
        showlegend=False,
        margin=dict(t=70, l=60, r=60, b=120),
    )
    return fig

def _trend_chart_pos_vs_lav(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    def _reg(df, col):
        if df.empty or col not in df.columns: return None
        y = df[col].astype(float).fillna(0.0).to_numpy()
        if np.allclose(y, 0): return None
        x_dates = pd.to_datetime(df["week_start"])
        x_num = x_dates.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
        if len(x_num) < 2: return None
        idx = np.argsort(x_num)
        x_num = x_num[idx]; y = y[idx]
        a, b = np.polyfit(x_num, y, 1)
        x0, x1 = x_num.min(), x_num.max()
        y0, y1 = a * x0 + b, a * x1 + b
        return pd.DataFrame({
            "x": [pd.Timestamp.fromordinal(int(x0)), pd.Timestamp.fromordinal(int(x1))],
            "y": [y0, y1],
        })
    rpos = _reg(df, "pos")
    if rpos is not None:
        fig.add_trace(go.Scatter(
            x=rpos["x"], y=rpos["y"],
            mode="lines",
            line=dict(width=3, color=GREEN),
            showlegend=False,
        ))
    rlav = _reg(df, "lav_gen")
    if rlav is not None:
        fig.add_trace(go.Scatter(
            x=rlav["x"], y=rlav["y"],
            mode="lines",
            line=dict(width=3, color=FUXIA),
            showlegend=False,
        ))
    fig.update_layout(
        title=dict(text="Trend settimanale: Positivi vs Lavorazione generale", pad=dict(t=6)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=520,
        xaxis_title="", yaxis_title="Valore",
        showlegend=False,
        margin=dict(t=70, l=60, r=60, b=120),
    )
    return fig

def _bar_mese(dfm: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=dfm["mese"],
        y=dfm["conf_pct"],
        marker_color=ORANGE,
        showlegend=False,
    )
    fig.add_bar(
        x=dfm["mese"],
        y=dfm["caduti_pct"],
        marker_color=CADUTI,
        showlegend=False,
    )
    fig.update_layout(
        title=dict(text="Ripartizione per mese (100% = positivi)", pad=dict(t=6)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        barmode="stack",
        barnorm="percent",
        height=520,
        yaxis_title="Percentuale",
        showlegend=False,
        margin=dict(t=70, l=60, r=60, b=120),
    )
    return fig

def _bar_operatori(dfo: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        x=dfo["operatore_short"],
        y=dfo["conf_pct"],
        marker_color=ORANGE,
        showlegend=False,
    )
    fig.add_bar(
        x=dfo["operatore_short"],
        y=dfo["caduti_pct"],
        marker_color=CADUTI,
        showlegend=False,
    )
    fig.update_layout(
        title=dict(text="Ripartizione per operatore (100% = positivi)", pad=dict(t=6)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        barmode="stack",
        barnorm="percent",
        height=520,
        yaxis_title="Percentuale",
        showlegend=False,
        margin=dict(t=70, l=60, r=60, b=120),
    )
    fig.update_xaxes(tickangle=-70)
    return fig

def _bar_lavorazioni_mese(dfm: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=dfm["mese"], y=dfm["pct_contatti"], marker_color=BLUE,   showlegend=False)
    fig.add_bar(x=dfm["mese"], y=dfm["pct_varie"],    marker_color=TEAL,   showlegend=False)
    fig.add_bar(x=dfm["mese"], y=dfm["pct_in_ch"],    marker_color=YELLOW, showlegend=False)
    fig.add_bar(x=dfm["mese"], y=dfm["pct_conv"],     marker_color=FUXIA,  showlegend=False)
    fig.add_bar(x=dfm["mese"], y=dfm["pct_attesa"],   marker_color=CADUTI, showlegend=False)
    fig.update_layout(
        title=dict(text="Lavorazioni per mese (100% = lavorazione generale)", pad=dict(t=6)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        barmode="stack", barnorm="percent",
        height=520,
        yaxis_title="Percentuale",
        showlegend=False,
        margin=dict(t=70, l=60, r=60, b=120),
    )
    return fig

def _bar_lavorazioni_operatori(dfo: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=dfo["operatore_short"], y=dfo["pct_contatti"], marker_color=BLUE,   showlegend=False)
    fig.add_bar(x=dfo["operatore_short"], y=dfo["pct_varie"],    marker_color=TEAL,   showlegend=False)
    fig.add_bar(x=dfo["operatore_short"], y=dfo["pct_in_ch"],    marker_color=YELLOW, showlegend=False)
    fig.add_bar(x=dfo["operatore_short"], y=dfo["pct_conv"],     marker_color=FUXIA,  showlegend=False)
    fig.add_bar(x=dfo["operatore_short"], y=dfo["pct_attesa"],   marker_color=CADUTI, showlegend=False)
    fig.update_layout(
        title=dict(text="Lavorazioni per operatore (100% = lavorazione generale)", pad=dict(t=6)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        barmode="stack", barnorm="percent",
        height=520,
        yaxis_title="Percentuale",
        showlegend=False,
        margin=dict(t=70, l=60, r=60, b=120),
    )
    fig.update_xaxes(tickangle=-70)
    return fig

def _bar_stack_mensile_per_operatore(df: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=title, pad=dict(t=6)),
            template="plotly_dark",
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            height=520,
            barmode="stack",
            barnorm="percent",
            showlegend=False,
            margin=dict(t=70, l=60, r=60, b=120),
        )
        return fig
    mesi = sorted(df["mese"].unique().tolist())
    ops = df["operatore_short"].fillna(df["operatore"]).unique().tolist()
    fig = go.Figure()
    for idx, op in enumerate(ops):
        col = OP_STACK_COLORS[idx % len(OP_STACK_COLORS)]
        vals = []
        for m in mesi:
            sub = df[df["mese"] == m]
            tot = sub[value_col].sum()
            op_val = sub.loc[sub["operatore_short"] == op, value_col].sum()
            pct = op_val if tot == 0 else (op_val / tot * 100.0)
            vals.append(pct)
        fig.add_bar(
            x=mesi,
            y=vals,
            marker_color=col,
            showlegend=False,
        )
    fig.update_layout(
        title=dict(text=title, pad=dict(t=6)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        barmode="stack",
        barnorm="percent",
        height=520,
        yaxis_title="Percentuale",
        showlegend=False,
        margin=dict(t=70, l=60, r=60, b=120),
    )
    return fig

# ================== POSITIVI / ORA ==================
def _positivi_per_ora_mese(schema_now: str, vkey: str,
                           mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    df_pos = _monthly_pos_conf(schema_now, vkey, mesi_sel, att_sel, op_sel)
    df_lav = _monthly_lavorazioni(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_pos.empty or df_lav.empty:
        return pd.DataFrame(columns=["mese","positivi_per_ora"])
    df = pd.merge(df_pos[["mese","positivi"]], df_lav[["mese","lav_gen"]], on="mese", how="inner")
    if df.empty:
        return pd.DataFrame(columns=["mese","positivi_per_ora"])
    df["positivi_per_ora"] = np.where(df["lav_gen"]>0, df["positivi"]/df["lav_gen"], 0)
    return df.sort_values("mese")

def _positivi_per_ora_operatore(schema_now: str, vkey: str,
                                mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    df_pos = _operator_pos_conf(schema_now, vkey, mesi_sel, att_sel, op_sel)
    df_lav = _operator_lavorazioni(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_pos.empty or df_lav.empty:
        return pd.DataFrame(columns=["operatore_short","positivi_per_ora"])
    df = pd.merge(
        df_pos[["operatore","operatore_short","positivi"]],
        df_lav[["operatore","operatore_short","lav_gen"]],
        on=["operatore","operatore_short"],
        how="inner"
    )
    if df.empty:
        return pd.DataFrame(columns=["operatore_short","positivi_per_ora"])
    df["positivi_per_ora"] = np.where(df["lav_gen"]>0, df["positivi"]/df["lav_gen"], 0)
    return df.sort_values("operatore")

# ================== PROCESSATI PER POSITIVO (CORRETTI) ==================
@st.cache_data(ttl=60, show_spinner=False)
def _proc_per_pos_mese(schema_now: str, vkey: str,
                       mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    rows = []
    tbls = _list_tables(schema_now, vkey)
    for _, r in tbls.iterrows():
        df = _load_table(schema_now, r["table_name"], ROW_LIMIT_PER_TABLE, vkey)
        if df.empty:
            continue
        dcol  = _find_col(df, ALIAS_DATA)
        pcol  = _find_col(df, ALIAS_POS)
        prcol = _find_col(df, ALIAS_PROC)
        cfp   = _find_col(df, ALIAS_POS_CONF)
        if not (dcol and pcol and prcol):
            continue
        acol = _find_col(df, ALIAS_ATT)
        ocol = _find_col(df, ALIAS_OP)
        dt = _to_datetime_fast(df[dcol])
        month = dt.dt.to_period("M").astype(str)
        mask = pd.Series(True, index=df.index)
        if mesi_sel:
            mask &= month.isin(mesi_sel)
        if att_sel and acol:
            mask &= df[acol].astype(str).isin(att_sel)
        if op_sel and ocol:
            mask &= df[ocol].astype(str).isin(op_sel)
        if not mask.any():
            continue
        pos  = _to_numeric_fast(df.loc[mask, pcol])
        proc = _to_numeric_fast(df.loc[mask, prcol])
        conf = _to_numeric_fast(df.loc[mask, cfp]) if cfp else pd.Series(0, index=df.index[mask], dtype=float)
        good = (pos > 0) & (proc > 0)
        if not good.any():
            continue
        tmp = pd.DataFrame({
            "mese": month[mask][good],
            "positivi": pos[good],
            "processati": proc[good],
            "positivi_conf": conf[good],
        })
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["mese","positivi","processati","positivi_conf","proc_per_pos"]).sort_values("mese")
    all_df = pd.concat(rows, ignore_index=True)
    agg = (all_df.groupby("mese", as_index=False)
           .agg({"positivi":"sum","processati":"sum","positivi_conf":"sum"}))
    agg["proc_per_pos"] = np.where(agg["positivi"]>0, agg["processati"]/agg["positivi"], 0)
    return agg.sort_values("mese")

@st.cache_data(ttl=60, show_spinner=False)
def _proc_per_pos_operatore(schema_now: str, vkey: str,
                            mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    rows = []
    tbls = _list_tables(schema_now, vkey)
    for _, r in tbls.iterrows():
        df = _load_table(schema_now, r["table_name"], ROW_LIMIT_PER_TABLE, vkey)
        if df.empty:
            continue
        ocol = _find_col(df, ALIAS_OP)
        pcol = _find_col(df, ALIAS_POS)
        prc  = _find_col(df, ALIAS_PROC)
        cfp  = _find_col(df, ALIAS_POS_CONF)
        if not (ocol and pcol and prc):
            continue
        dcol = _find_col(df, ALIAS_DATA)
        acol = _find_col(df, ALIAS_ATT)
        mask = pd.Series(True, index=df.index)
        if dcol:
            dt = _to_datetime_fast(df[dcol])
            month = dt.dt.to_period("M").astype(str)
            if mesi_sel:
                mask &= month.isin(mesi_sel)
        if att_sel and acol:
            mask &= df[acol].astype(str).isin(att_sel)
        if op_sel:
            mask &= df[ocol].astype(str).isin(op_sel)
        if not mask.any():
            continue
        pos  = _to_numeric_fast(df.loc[mask, pcol])
        proc = _to_numeric_fast(df.loc[mask, prc])
        conf = _to_numeric_fast(df.loc[mask, cfp]) if cfp else pd.Series(0, index=df.index[mask], dtype=float)
        good = (pos > 0) & (proc > 0)
        if not good.any():
            continue
        tmp = pd.DataFrame({
            "operatore": df.loc[mask, ocol][good].astype(str),
            "operatore_short": df.loc[mask, ocol][good].astype(str).map(_short_name),
            "positivi": pos[good],
            "processati": proc[good],
            "positivi_conf": conf[good],
        })
        rows.append(tmp)
    if not rows:
        return pd.DataFrame(columns=["operatore","operatore_short","positivi","processati","positivi_conf","proc_per_pos"]).sort_values("operatore")
    all_df = pd.concat(rows, ignore_index=True)
    agg = (all_df.groupby(["operatore","operatore_short"], as_index=False)
           .agg({"positivi":"sum","processati":"sum","positivi_conf":"sum"}))
    agg["proc_per_pos"] = np.where(agg["positivi"]>0, agg["processati"]/agg["positivi"], 0)
    return agg.sort_values("operatore")

# ================== RED% (da processati veri) ==================
@st.cache_data(ttl=60, show_spinner=False)
def _red_per_mese(schema_now: str, vkey: str,
                  mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    df_pp = _proc_per_pos_mese(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_pp.empty:
        return pd.DataFrame(columns=["mese","red_pct","red_conf_pct"])
    df_pp["red_pct"] = np.where(df_pp["processati"]>0, df_pp["positivi"]/df_pp["processati"]*100, 0)
    df_pp["red_conf_pct"] = np.where(df_pp["processati"]>0, df_pp["positivi_conf"]/df_pp["processati"]*100, 0)
    return df_pp[["mese","red_pct","red_conf_pct"]].sort_values("mese")

@st.cache_data(ttl=60, show_spinner=False)
def _red_per_operatore(schema_now: str, vkey: str,
                       mesi_sel: List[str], att_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    df_pp = _proc_per_pos_operatore(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_pp.empty:
        return pd.DataFrame(columns=["operatore","operatore_short","red_pct","red_conf_pct"])
    df_pp["red_pct"] = np.where(df_pp["processati"]>0, df_pp["positivi"]/df_pp["processati"]*100, 0)
    df_pp["red_conf_pct"] = np.where(df_pp["processati"]>0, df_pp["positivi_conf"]/df_pp["processati"]*100, 0)
    return df_pp[["operatore","operatore_short","red_pct","red_conf_pct"]].sort_values("operatore")

# ================== TOTALE ORE LAVORAZIONE GENERALE ==================
def _tot_lav_mese(df: pd.DataFrame) -> pd.DataFrame:
    return df[["mese","lav_gen"]].sort_values("mese")

def _tot_lav_operatore(df: pd.DataFrame) -> pd.DataFrame:
    return df[["operatore","operatore_short","lav_gen"]].sort_values("operatore")

# ================== BAR SEMPLICE CON MEAN/TEAM ==================
def _bar_simple_with_mean(
    x: List[str],
    y: List[float],
    title: str,
    y_title: str,
    is_operator: bool = False,
    color: str = GREEN,
) -> go.Figure:
    fig = go.Figure()
    x = list(x)
    y = list(y)
    if not x:
        fig.update_layout(
            title=dict(text=title, pad=dict(t=6)),
            template="plotly_dark",
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            height=460,
            yaxis_title=y_title,
            showlegend=False,
            margin=dict(t=70, l=60, r=60, b=120),
        )
        return fig
    mean_val = float(np.mean(y)) if y else 0.0
    extra_name = "Team" if is_operator else "Mean"
    x_all = x + [extra_name]
    y_all = y + [mean_val]
    fig.add_bar(
        x=x_all,
        y=y_all,
        marker_color=[color]*len(x) + ["#64748b"],
        showlegend=False,
    )
    upper = mean_val * 1.05
    lower = mean_val * 0.95
    fig.add_trace(go.Scatter(
        x=[x_all[0], x_all[-1]],
        y=[upper, upper],
        mode="lines",
        line=dict(width=0),
        fill=None,
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[x_all[0], x_all[-1]],
        y=[lower, lower],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=GREY_OVERLAY,
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_all,
        y=[mean_val]*len(x_all),
        mode="lines",
        line=dict(color=VIOLET, width=2),
        showlegend=False,
    ))
    if all(isinstance(xx, str) and re.match(r"^\d{4}-\d{2}$", xx) for xx in x):
        try:
            months_dt = pd.to_datetime([f"{xx}-01" for xx in x], errors="coerce")
            x_num = months_dt.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
            y_arr = np.array(y, dtype=float)
            if len(x_num) >= 2 and not np.allclose(y_arr, 0):
                idx = np.argsort(x_num)
                x_sorted = [x[i] for i in idx]
                x_num = x_num[idx]; y_arr = y_arr[idx]
                a, b = np.polyfit(x_num, y_arr, 1)
                x0, x1 = x_num.min(), x_num.max()
                y0, y1 = a * x0 + b, a * x1 + b
                fig.add_trace(go.Scatter(
                    x=[x_sorted[0], x_sorted[-1]],
                    y=[y0, y1],
                    mode="lines+markers",
                    line=dict(width=3, color=ORANGE),
                    marker=dict(size=6, color=ORANGE),
                    showlegend=False,
                ))
        except Exception:
            pass
    fig.update_layout(
        title=dict(text=title, pad=dict(t=6)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=460,
        yaxis_title=y_title,
        showlegend=False,
        margin=dict(t=70, l=60, r=60, b=120),
    )
    if is_operator:
        fig.update_xaxes(tickangle=-70)
    return fig

# ================== HTML REPORT ==================
def _build_html_report(schema_now: str, vkey: str, filters: dict) -> str:
    head = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>Performances & KPIs</title>
<style>
body {{ font-family: Arial, sans-serif; color:#111; }}
h1,h2,h3 {{ margin: 8px 0; }}
.section {{ page-break-inside: avoid; margin: 18px 0; }}
table {{ border-collapse: collapse; width:100%; font-size:12px; }}
th,td {{ border:1px solid #ddd; padding:6px; }}
.muted {{ color:#444; }}
</style></head><body>
<h1>Performances &amp; KPIs</h1>
<p class="muted">Schema: <b>{schema_now}</b> | vkey: {vkey}</p>
<p class="muted">Filtri: {filters}</p>
"""
    tbls = _list_tables(schema_now, vkey)
    body = []
    body.append("<div class='section'><h2>Tabelle nello schema</h2>")
    body.append(tbls[["table_name","table_type","table_rows"]].to_html(index=False))
    body.append("</div>")
    tail = "</body></html>"
    return head + "\n".join(body) + tail

# ================== PAGINA ==================
def render() -> None:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60_000, key="sdr_autorefresh")
    except Exception:
        pass

    schema_now = _current_db()
    vkey = f"{_db_version(schema_now)}|{int(time.time()//60)}"

    st.title("Performances & KPIs")
    st.caption(f"Schema attivo: {schema_now}  |  vkey: {vkey}")

    mesi_opts, att_opts, op_opts = _collect_filter_options(schema_now, vkey)
    c1, c2, c3 = st.columns(3)
    with c1:
        mesi_sel = st.multiselect("Mese (YYYY-MM)", mesi_opts, default=[], key="sdr_mesi")
    with c2:
        att_sel = st.multiselect("Attività", att_opts, default=[], key="sdr_att")
    with c3:
        op_sel = st.multiselect("Operatore", op_opts, default=[], key="sdr_op")

    # 1) Linea settimanale
    st.markdown("### Linea settimanale: Ore conversazione vs Positivi")
    dfw = _weekly_conv_positivi_filtered(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if dfw.empty:
        st.warning("Nessun dato utile trovato con i filtri correnti.")
    else:
        st.plotly_chart(_weekly_chart(dfw), use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_linea_sett_html(), unsafe_allow_html=True)
        st.caption(
            f"Settimane: {len(dfw)}  |  Tot ore: {dfw['Ore conversazione'].sum():.2f}  |  Tot positivi: {int(dfw['Positivi'].sum())}"
        )

    # 2) Trend settimanale (rette principali)
    st.markdown("### Trend settimanale (rette)")
    dft = _weekly_for_trend(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if dft.empty:
        st.info("Dati insufficienti per calcolare le rette di trend.")
    else:
        st.plotly_chart(_trend_chart_lines(dft), use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_trend_sett_html(), unsafe_allow_html=True)

    # 2B) Trend settimanale: Positivi vs Lavorazione generale
    dft_lav = _weekly_for_trend_lav(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if not dft_lav.empty:
        st.plotly_chart(_trend_chart_pos_vs_lav(dft_lav), use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_trend_lav_html(), unsafe_allow_html=True)
    else:
        st.info("Nessuna colonna di lavorazione generale trovata per costruire il trend settimanale.")

    # 3) Barre per mese (positivi)
    st.markdown("### Distribuzione per mese (100% positivi)")
    dfm = _monthly_pos_conf(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if dfm.empty:
        st.info("Nessun dato mensile da mostrare.")
    else:
        st.plotly_chart(_bar_mese(dfm), use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_bar_mese_html(), unsafe_allow_html=True)

    # 4) Barre per operatore (positivi)
    st.markdown("### Distribuzione per operatore (100% positivi)")
    dfo = _operator_pos_conf(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if dfo.empty:
        st.info("Nessun dato per operatore da mostrare.")
    else:
        st.plotly_chart(_bar_operatori(dfo), use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_bar_mese_html(), unsafe_allow_html=True)

    # 5) Lavorazioni per mese
    st.markdown("### Lavorazioni per mese")
    df_lav_m = _monthly_lavorazioni(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_lav_m.empty:
        st.info("Nessuna colonna di lavorazione generale trovata per mese.")
    else:
        show_df(df_lav_m[["mese","lav_gen","lav_contatti","lav_varie","in_chiamata","conversazione","in_attesa"]])
        st.plotly_chart(_bar_lavorazioni_mese(df_lav_m), use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_lavorazioni_html(), unsafe_allow_html=True)

    # 6) Lavorazioni per operatore
    st.markdown("### Lavorazioni per operatore")
    df_lav_o = _operator_lavorazioni(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_lav_o.empty:
        st.info("Nessuna colonna di lavorazione generale trovata per operatore.")
    else:
        show_df(df_lav_o[["operatore","lav_gen","lav_contatti","lav_varie","in_chiamata","conversazione","in_attesa"]])
        st.plotly_chart(_bar_lavorazioni_operatori(df_lav_o), use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_lavorazioni_html(), unsafe_allow_html=True)

    # 7) Positivi per ora di lavoro (per mese)
    st.markdown("### Positivi per ora di lavoro (per mese)")
    df_pos_ora_m = _positivi_per_ora_mese(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_pos_ora_m.empty:
        st.info("Nessun dato sufficiente per calcolare positivi/ora per mese.")
    else:
        show_df(df_pos_ora_m)
        fig = _bar_simple_with_mean(
            x=df_pos_ora_m["mese"].tolist(),
            y=df_pos_ora_m["positivi_per_ora"].tolist(),
            title="Positivi per ora di lavoro (per mese)",
            y_title="Positivi/ora",
            is_operator=False,
            color=GREEN,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_bar_simple_html(has_trend=True), unsafe_allow_html=True)

    # 8) Positivi per ora di lavoro (per operatore)
    st.markdown("### Positivi per ora di lavoro (per operatore)")
    df_pos_ora_o = _positivi_per_ora_operatore(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_pos_ora_o.empty:
        st.info("Nessun dato sufficiente per calcolare positivi/ora per operatore.")
    else:
        show_df(df_pos_ora_o[["operatore","operatore_short","positivi","lav_gen","positivi_per_ora"]])
        fig = _bar_simple_with_mean(
            x=df_pos_ora_o["operatore_short"].tolist(),
            y=df_pos_ora_o["positivi_per_ora"].tolist(),
            title="Positivi per ora di lavoro (per operatore)",
            y_title="Positivi/ora",
            is_operator=True,
            color=GREEN,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_bar_simple_html(has_trend=False), unsafe_allow_html=True)

    # 9) Rapporto In chiamata / Lavorazione generale (per mese)
    st.markdown("### Rapporto In chiamata / Lavorazione generale (per mese)")
    df_inch_m = _inch_lav_mese(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_inch_m.empty:
        st.info("Nessun dato disponibile per il rapporto mensile.")
    else:
        show_df(df_inch_m)
        fig = _bar_simple_with_mean(
            x=df_inch_m["mese"].tolist(),
            y=df_inch_m["ratio_pct"].tolist(),
            title="Rapporto In chiamata / Lavorazione generale (per mese)",
            y_title="%",
            is_operator=False,
            color=ORANGE,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_bar_simple_html_orange(has_trend=True), unsafe_allow_html=True)

    # 10) Rapporto In chiamata / Lavorazione generale (per operatore)
    st.markdown("### Rapporto In chiamata / Lavorazione generale (per operatore)")
    df_inch_o = _inch_lav_operatore(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_inch_o.empty:
        st.info("Nessun dato disponibile per il rapporto per operatore.")
    else:
        show_df(df_inch_o[["operatore","operatore_short","in_chiamata","lav_gen","ratio_pct"]])
        fig = _bar_simple_with_mean(
            x=df_inch_o["operatore_short"].tolist(),
            y=df_inch_o["ratio_pct"].tolist(),
            title="Rapporto In chiamata / Lavorazione generale (per operatore)",
            y_title="%",
            is_operator=True,
            color=ORANGE,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_bar_simple_html_orange(has_trend=False), unsafe_allow_html=True)

    # 11) Processati per positivo (per mese)
    st.markdown("### Processati per positivo (per mese)")
    df_pp_m = _proc_per_pos_mese(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_pp_m.empty:
        st.info("Nessun dato per processati/positivo per mese.")
    else:
        show_df(df_pp_m)
        fig = _bar_simple_with_mean(
            x=df_pp_m["mese"].tolist(),
            y=df_pp_m["proc_per_pos"].tolist(),
            title="Processati per positivo (per mese)",
            y_title="Processati / positivo",
            is_operator=False,
            color="#22c55e",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_bar_simple_html_proc(has_trend=True), unsafe_allow_html=True)

    # 12) Processati per positivo (per operatore)
    st.markdown("### Processati per positivo (per operatore)")
    df_pp_o = _proc_per_pos_operatore(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_pp_o.empty:
        st.info("Nessun dato per processati/positivo per operatore.")
    else:
        show_df(df_pp_o[["operatore","operatore_short","processati","positivi","positivi_conf","proc_per_pos"]])
        fig = _bar_simple_with_mean(
            x=df_pp_o["operatore_short"].tolist(),
            y=df_pp_o["proc_per_pos"].tolist(),
            title="Processati per positivo (per operatore)",
            y_title="Processati / positivo",
            is_operator=True,
            color="#22c55e",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_bar_simple_html_proc(has_trend=False), unsafe_allow_html=True)

    # 13) Totale ore lavorazione generale (per mese)
    st.markdown("### Totale ore lavorazione generale (per mese)")
    if df_lav_m.empty:
        st.info("Nessun dato di lavorazione generale per mese.")
    else:
        df_tot_m = _tot_lav_mese(df_lav_m)
        fig = _bar_simple_with_mean(
            x=df_tot_m["mese"].tolist(),
            y=df_tot_m["lav_gen"].tolist(),
            title="Totale ore di lavorazione generale (per mese)",
            y_title="Ore",
            is_operator=False,
            color=BLUE,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_bar_simple_html_blue(has_trend=True), unsafe_allow_html=True)

    # 14) Totale ore lavorazione generale (per operatore)
    st.markdown("### Totale ore lavorazione generale (per operatore)")
    if df_lav_o.empty:
        st.info("Nessun dato di lavorazione generale per operatore.")
    else:
        df_tot_o = _tot_lav_operatore(df_lav_o)
        fig = _bar_simple_with_mean(
            x=df_tot_o["operatore_short"].tolist(),
            y=df_tot_o["lav_gen"].tolist(),
            title="Totale ore di lavorazione generale (per operatore)",
            y_title="Ore",
            is_operator=True,
            color=BLUE,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_bar_simple_html_blue(has_trend=False), unsafe_allow_html=True)

    # 14B) NUOVO: Distribuzione mensile lavorazione generale per operatore (100%)
    st.markdown("### Distribuzione mensile lavorazione generale per operatore (100%)")
    df_mop = _monthly_operator_lavorazioni(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_mop.empty:
        st.info("Nessun dato per distribuzione mensile per operatore.")
    else:
        fig = _bar_stack_mensile_per_operatore(
            df_mop, "lav_gen",
            "Distribuzione mensile lavorazione generale per operatore (100% = tot mese)"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        ops = df_mop["operatore_short"].fillna(df_mop["operatore"]).unique().tolist()
        cols = [OP_STACK_COLORS[i % len(OP_STACK_COLORS)] for i in range(len(ops))]
        st.markdown(_legend_dist_stack_html(ops, cols, "Legenda operatori"), unsafe_allow_html=True)

    # 14C) NUOVO: Distribuzione mensile IN CHIAMATA per operatore (100%)
    st.markdown("### Distribuzione mensile IN CHIAMATA per operatore (100%)")
    if df_mop.empty:
        st.info("Nessun dato per distribuzione IN CHIAMATA per operatore.")
    else:
        fig = _bar_stack_mensile_per_operatore(
            df_mop, "in_chiamata",
            "Distribuzione mensile IN CHIAMATA per operatore (100% = tot mese)"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        ops = df_mop["operatore_short"].fillna(df_mop["operatore"]).unique().tolist()
        cols = [OP_STACK_COLORS[i % len(OP_STACK_COLORS)] for i in range(len(ops))]
        st.markdown(_legend_dist_stack_html(ops, cols, "Legenda operatori"), unsafe_allow_html=True)

    # 15) RED% (per mese)
    st.markdown("### RED% (per mese)")
    df_red_m = _red_per_mese(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_red_m.empty:
        st.info("Nessun dato per RED% per mese.")
    else:
        show_df(df_red_m)
        fig = go.Figure()
        fig.add_bar(
            x=df_red_m["mese"],
            y=df_red_m["red_pct"],
            marker_color="#0ea5e9",
            showlegend=False,
        )
        fig.add_bar(
            x=df_red_m["mese"],
            y=df_red_m["red_conf_pct"],
            marker_color="#f97316",
            showlegend=False,
        )
        try:
            mesi = df_red_m["mese"].tolist()
            mesi_dt = pd.to_datetime([f"{m}-01" for m in mesi], errors="coerce")
            x_num = mesi_dt.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
            y1 = df_red_m["red_pct"].to_numpy(dtype=float)
            y2 = df_red_m["red_conf_pct"].to_numpy(dtype=float)
            if len(x_num) >= 2 and not np.allclose(y1, 0):
                idx = np.argsort(x_num)
                x_sorted = [mesi[i] for i in idx]
                x_num1 = x_num[idx]; y1s = y1[idx]
                a1, b1 = np.polyfit(x_num1, y1s, 1)
                x0, x1v = x_num1.min(), x_num1.max()
                y0, y1v = a1 * x0 + b1, a1 * x1v + b1
                fig.add_trace(go.Scatter(
                    x=[x_sorted[0], x_sorted[-1]],
                    y=[y0, y1v],
                    mode="lines+markers",
                    line=dict(width=2, color="#0ea5e9"),
                    showlegend=False,
                ))
            if len(x_num) >= 2 and not np.allclose(y2, 0):
                idx = np.argsort(x_num)
                x_sorted = [mesi[i] for i in idx]
                x_num2 = x_num[idx]; y2s = y2[idx]
                a2, b2 = np.polyfit(x_num2, y2s, 1)
                x0, x1v = x_num2.min(), x_num2.max()
                y0, y1v = a2 * x0 + b2, a2 * x1v + b2
                fig.add_trace(go.Scatter(
                    x=[x_sorted[0], x_sorted[-1]],
                    y=[y0, y1v],
                    mode="lines+markers",
                    line=dict(width=2, color="#f97316"),
                    showlegend=False,
                ))
        except Exception:
            pass
        fig.update_layout(
            title=dict(text="RED% (per mese)", pad=dict(t=6)),
            template="plotly_dark",
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            barmode="group",
            height=520,
            yaxis_title="%",
            showlegend=False,
            margin=dict(t=70, l=60, r=60, b=120),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_red_mese_html(), unsafe_allow_html=True)

    # 16) RED% (per operatore)
    st.markdown("### RED% (per operatore)")
    df_red_o = _red_per_operatore(schema_now, vkey, mesi_sel, att_sel, op_sel)
    if df_red_o.empty:
        st.info("Nessun dato per RED% per operatore.")
    else:
        show_df(df_red_o[["operatore","operatore_short","red_pct","red_conf_pct"]])
        x_ops = df_red_o["operatore_short"].tolist()
        red1 = df_red_o["red_pct"].tolist()
        red2 = df_red_o["red_conf_pct"].tolist()
        mean1 = float(np.mean(red1)) if red1 else 0.0
        mean2 = float(np.mean(red2)) if red2 else 0.0
        x_all = x_ops + ["Team"]
        red1_all = red1 + [mean1]
        red2_all = red2 + [mean2]
        fig = go.Figure()
        fig.add_bar(
            x=x_all,
            y=red1_all,
            marker_color="#0ea5e9",
            showlegend=False,
        )
        fig.add_bar(
            x=x_all,
            y=red2_all,
            marker_color="#f97316",
            showlegend=False,
        )
        fig.add_trace(go.Scatter(
            x=x_all,
            y=[mean1]*len(x_all),
            mode="lines",
            line=dict(width=2, color="#0ea5e9", dash="dot"),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x_all,
            y=[mean2]*len(x_all),
            mode="lines",
            line=dict(width=2, color="#f97316", dash="dot"),
            showlegend=False,
        ))
        fig.update_layout(
            title=dict(text="RED% (per operatore)", pad=dict(t=6)),
            template="plotly_dark",
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            barmode="group",
            height=520,
            yaxis_title="%",
            showlegend=False,
            margin=dict(t=70, l=60, r=60, b=120),
        )
        fig.update_xaxes(tickangle=-70)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_red_operatore_html(), unsafe_allow_html=True)

    # elenco tabelle
    tbls = _list_tables(schema_now, vkey)
    st.caption(f"Tabelle nello schema `{schema_now}`: {len(tbls)}")
    show_df(tbls[["table_name","table_type","table_rows"]], height=280)

    # bottoni
    st.markdown("""
    <style>
      div[data-testid="stDownloadButton"] > button,
      div[data-testid="stButton"] > button {
        background: #ff00ff !important;
        color: #0b0b0b !important;
        border: none !important;
        border-radius: 18px !important;
        padding: 22px 18px !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        width: 100% !important;
        height: 86px !important;
        box-shadow: none !important;
      }
    </style>
    """, unsafe_allow_html=True)

    filters_dict = {"mesi": mesi_sel, "attivita": att_sel, "operatori": op_sel}
    html_report = _build_html_report(schema_now, vkey, filters_dict)
    buf = io.BytesIO(html_report.encode("utf-8"))

    c1b, c2b = st.columns(2)
    with c1b:
        st.download_button(
            "⬇️ Scarica HTML",
            data=buf,
            file_name=f"SDR_report_{schema_now}_{int(time.time())}.html",
            mime="text/html",
            key="sdr_html",
            use_container_width=True,
        )
    with c2b:
        if st.button("🏠 Home", key="sdr_home", use_container_width=True):
            go_home()

if __name__ == "__main__":
    render()
