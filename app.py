#!/usr/bin/env python3
# app.py — Home + routing (con Attività TLK e Scheda Operatore)
# Aggiornato: data loader integrato (CSV ';' preferred / dataset.py / dataset_embedded)
# Includes automatic semicolon CSV fix and light normalization helpers.

from __future__ import annotations

import streamlit as st
from pathlib import Path
import importlib
import pandas as pd
import sys
import shutil
import re
import time

st.set_page_config(page_title="AEDB", page_icon="📊", layout="wide")

SMART_FUXIA = "#FF00FF"
HOVER_PURPLE = "#7A00FF"
CHECK_GREEN = "#00FF66"

# ================= CSS globale minimale =================
st.markdown(
    f"""
    <style>
      html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"],
      .block-container, [data-testid="stSidebar"], .main {{
        background:#000 !important; padding:0 !important; margin:0 !important; height:auto !important;
      }}
      [data-testid="stHeader"], header, footer, #MainMenu {{ display:none !important; }}
      .block-container, .stApp, .main {{ padding-top:0 !important; margin-top:0 !important; }}

      .title-wrap {{ height:96px; display:flex; align-items:center; justify-content:center; }}
      .title {{ color:{SMART_FUXIA}; font-size:64px; font-weight:900; letter-spacing:2px; margin:0; }}

      div.stButton > button {{
        width:270px !important; height:105px !important;
        background:{SMART_FUXIA}; color:#000; border:none; border-radius:12px;
        cursor:pointer; font-weight:800; font-size:18px; margin:0 12px; white-space:pre-line;
        transition: background-color .15s ease, transform .08s ease;
      }}
      div.stButton > button:hover  {{ background:{HOVER_PURPLE}; color:#fff; }}

      input[type="checkbox"], .stCheckbox input[type="checkbox"] {{ accent-color:{CHECK_GREEN} !important; }}

      .footer-fixed {{
        position:fixed; left:0; right:0; bottom:0; height:48px;
        display:flex; align-items:center; justify-content:center;
        color:{SMART_FUXIA}; font-weight:700; background:#000; z-index:9;
        border-top:1px solid rgba(255,255,255,0.05);
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ================ DATA LOADER =================
PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DESKTOP_CSV_DIR = Path(r"C:\Users\HP\Desktop\DATABASE_TLK")

tables: dict[str, pd.DataFrame] = {}  # original, fixed tables
_norm_cache: dict[str, pd.DataFrame] = {}  # normalized copies (lazy)

def _fix_semicolon_single_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] != 1:
        return df
    s = df.iloc[:, 0].astype(str).fillna('')
    sample_frac = min(30, len(s))
    sample = s.head(sample_frac).astype(str)
    if (sample.str.contains(';').sum() / max(1, len(sample))) < 0.4:
        return df
    parts = s.str.split(';', expand=True)
    first_row = parts.iloc[0].astype(str).tolist()
    header_keywords = ('attivita','data','operatore','lavorazione','positivi','processati','mese')
    if any(any(k in str(cell).lower() for k in header_keywords) for cell in first_row):
        parts.columns = [str(c).strip() for c in first_row]
        parts = parts.iloc[1:].reset_index(drop=True)
    return parts

def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        c2 = str(c)
        c2 = c2.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        c2 = c2.strip().strip('\ufeff').strip('"').strip("'")
        c2 = re.sub(r'\s+', ' ', c2)
        new_cols.append(c2)
    df.columns = new_cols
    return df

def _clean_string_cells(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=['object']).columns:
        try:
            df[c] = df[c].astype(str).str.strip().replace({'^nan$': None, '^None$': None}, regex=True)
            df[c] = df[c].str.replace(r'\s+', ' ', regex=True)
        except Exception:
            pass
    return df

def _ensure_mese_column(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [c for c in df.columns if re.search(r'data|date|mese|timestamp|giorno', c, flags=re.I)]
    for cand in candidates:
        try:
            series = pd.to_datetime(df[cand], errors='coerce', infer_datetime_format=True)
            if series.notna().sum() > 0:
                df['mese'] = series.dt.strftime('%Y-%m')
                return df
        except Exception:
            continue
    for c in df.columns:
        try:
            s = df[c].astype(str).str.extract(r'(\d{4}-\d{2})')[0]
            if s.notna().sum() > 0:
                df['mese'] = s
                return df
        except Exception:
            continue
    return df

def _postprocess_table(df: pd.DataFrame) -> pd.DataFrame:
    df = _fix_semicolon_single_column(df)
    df = _clean_headers(df)
    df = _clean_string_cells(df)
    df = _ensure_mese_column(df)
    return df

def _load_from_dataset_embedded():
    try:
        import dataset_embedded as de  # type: ignore
        names = de.list_tables()
        for n in names:
            df = de.load_table(n)
            df = _postprocess_table(df)
            tables[n] = df
        return True
    except Exception:
        return False

def _load_from_dataset_py():
    try:
        import dataset  # type: ignore
        try:
            loaded = getattr(dataset, "load_all_tables_from_embedded", lambda: {})()
            if loaded:
                for k, v in loaded.items():
                    tables[k] = _postprocess_table(v)
                return True
        except Exception:
            pass
        if hasattr(dataset, "tables") and isinstance(dataset.tables, dict) and dataset.tables:
            for k, v in dataset.tables.items():
                tables[k] = _postprocess_table(v)
            return True
        return False
    except Exception:
        return False

def _load_from_csv_folder(folder: Path):
    ok = False
    if not folder.exists():
        return False
    csvs = sorted(folder.glob("*.csv"))
    for p in csvs:
        name = p.stem
        try:
            df = pd.read_csv(p)
        except Exception:
            try:
                df = pd.read_csv(p, sep=';', engine='python')
            except Exception:
                df = pd.read_csv(p, engine='python')
        df = _postprocess_table(df)
        tables[name] = df
        ok = True
    return ok

def _load_from_csv_folder_prefer_semicolon(folder: Path):
    ok = False
    if not folder.exists():
        return False
    csvs = sorted(folder.glob("*.csv"))
    for p in csvs:
        name = p.stem
        df_loaded = None
        for sep in [';', ',', '\t', '|']:
            try:
                df_try = pd.read_csv(p, sep=sep, engine='python')
                if df_try.shape[1] > 1 or df_try.shape[0] > 0:
                    df_loaded = df_try
                    break
            except Exception:
                continue
        if df_loaded is not None:
            df_loaded = _postprocess_table(df_loaded)
            tables[name] = df_loaded
            ok = True
    return ok

# Run loaders in order: prefer CSVs (semicolon) first
loaded = False
loader_source = "none"
if _load_from_csv_folder_prefer_semicolon(PROJECT_DIR / "data"):
    loaded = True
    loader_source = "project data/ (./data) sep=';' preferred"
elif _load_from_dataset_py():
    loaded = True
    loader_source = "dataset.py"
elif _load_from_dataset_embedded():
    loaded = True
    loader_source = "dataset_embedded.py"
elif _load_from_csv_folder(DESKTOP_CSV_DIR):
    loaded = True
    loader_source = f"desktop {DESKTOP_CSV_DIR}"
    try:
        for p in sorted(DESKTOP_CSV_DIR.glob("*.csv")):
            dst = DATA_DIR / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
    except Exception:
        pass

# Helper: get original fixed table
def get_table(name: str) -> pd.DataFrame:
    if name in tables:
        return tables[name]
    raise KeyError(f"Table '{name}' not found. Available: {list(tables.keys())}")

# Helper: get normalized table (lowercase columns, stripped strings, mese ensured)
def get_table_norm(name: str) -> pd.DataFrame:
    if name in _norm_cache:
        return _norm_cache[name]
    if name not in tables:
        raise KeyError(f"Table '{name}' not found. Available: {list(tables.keys())}")
    df = tables[name].copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in df.select_dtypes(include=['object']).columns:
        try:
            df[c] = df[c].astype(str).str.strip().replace({'^nan$': None}, regex=True)
        except Exception:
            pass
    if 'mese' not in df.columns:
        df = _ensure_mese_column(df)
    _norm_cache[name] = df
    return df

# Put table names into session state for UI use
st.session_state.setdefault("tables_available", list(tables.keys()))
st.session_state.setdefault("data_loader_source", loader_source)

# ================ Safe dynamic imports of pages =================
def _safe_import(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None

# Try import modules, but do not fail if absent.
page_modules = {
    "eda": _safe_import("eda"),
    "sdr": _safe_import("SDR"),
    "sdrml": _safe_import("SDRML"),
    "sdrforecast": _safe_import("SDRforecast"),
    "statistica": _safe_import("Stats"),
    "probabilita": _safe_import("probability"),
    "attivita": _safe_import("atts"),
    "operatore": _safe_import("operatore"),
}

# ================ Router / UI helpers =================
if "page" not in st.session_state:
    st.session_state.page = "home"

def goto(p: str):
    """
    Robust page switch + rerun:
      - set session_state.page
      - try st.experimental_rerun()
      - fallback to st.rerun()
      - final fallback: set flag and st.stop()
    This avoids AttributeError on runtimes where experimental_rerun is missing.
    """
    st.session_state.page = p
    # try experimental rerun then rerun
    for fn in ("experimental_rerun", "rerun"):
        try:
            f = getattr(st, fn, None)
            if callable(f):
                f()
                return
        except Exception:
            # continue to next option
            continue
    # final fallback: mark need for rerun and stop execution cleanly
    st.session_state._need_rerun = True
    # st.stop will end execution for this run. The UI will reflect session_state change on next rerun.
    try:
        st.stop()
    except Exception:
        # if even st.stop fails, raise a clear error so logs show something actionable
        raise RuntimeError("Unable to rerun streamlit. Rerun methods not available in this runtime.")

def page_frame(_title: str, body_fn):
    try:
        body_fn()
    except Exception as e:
        st.exception(e)

# ================= Home =================
def render_home():
    st.markdown('<div class="title-wrap"><div class="title">TLK</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="home-spacer"></div>', unsafe_allow_html=True)

    cols_info = st.columns([1,2,2,2])
    with cols_info[0]:
        st.markdown(f"**Schema attivo:** aedbdata")
        st.write("")
    with cols_info[1]:
        st.markdown(f"**Dataloader:** `{st.session_state.data_loader_source}`")
    with cols_info[2]:
        st.markdown(f"**Tabelle disponibili:** {len(st.session_state.tables_available)}")
    with cols_info[3]:
        names_preview = st.session_state.tables_available[:6] or []
        st.markdown(" ".join(f"`{n}`" for n in names_preview))

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        if st.button("EDA\nExplorative Data Analysis"): goto("eda")
    with c2:
        if st.button("SDR Team\nSales Development Team Performances"): goto("sdr")
    with c3:
        if st.button("SDR - ML Insights"): goto("sdrml")
    with c4:
        if st.button("SDR Forecast"): goto("sdrforecast")

    r1, r2, r3, r4 = st.columns(4, gap="small")
    with r1:
        if st.button("Statistica"): goto("statistica")
    with r2:
        if st.button("Probabilità"): goto("probabilita")
    with r3:
        if st.button("Attività Clienti"): goto("attivita")
    with r4:
        if st.button("Scheda Operatore"): goto("operatore")

    st.markdown('<div class="footer-fixed">Dashboard and Data Analysis developed with Python and AI</div>',
                unsafe_allow_html=True)

# ================= Switch =================
p = st.session_state.page

if p == "home":
    render_home()

elif p == "eda":
    mod = page_modules.get("eda")
    if mod:
        try:
            setattr(mod, "tables", tables)
            setattr(mod, "get_table", get_table)
            setattr(mod, "get_table_norm", get_table_norm)
        except Exception:
            pass
        page_frame("Explorative Data Analysis", getattr(mod, "render", lambda: st.error("render() missing")))
    else:
        st.error("Modulo EDA non disponibile.")

elif p == "sdr":
    mod = page_modules.get("sdr")
    if mod:
        try:
            setattr(mod, "tables", tables)
            setattr(mod, "get_table", get_table)
            setattr(mod, "get_table_norm", get_table_norm)
        except Exception:
            pass
        page_frame("SDR Team", getattr(mod, "render", lambda: st.error("render() missing")))
    else:
        st.error("Modulo SDR non disponibile.")

elif p == "sdrml":
    mod = page_modules.get("sdrml")
    if mod:
        try:
            setattr(mod, "tables", tables)
            setattr(mod, "get_table", get_table)
            setattr(mod, "get_table_norm", get_table_norm)
        except Exception:
            pass
        page_frame("SDR Insight ML", getattr(mod, "render", lambda: st.error("render() missing")))
    else:
        st.error("Modulo SDRML non disponibile.")

elif p == "sdrforecast":
    mod = page_modules.get("sdrforecast")
    if mod:
        try:
            setattr(mod, "tables", tables)
            setattr(mod, "get_table", get_table)
            setattr(mod, "get_table_norm", get_table_norm)
        except Exception:
            pass
        page_frame("SDR Forecast", getattr(mod, "render", lambda: st.error("render() missing")))
    else:
        st.error("Modulo SDRforecast non disponibile.")

elif p == "statistica":
    mod = page_modules.get("statistica")
    if mod:
        try:
            setattr(mod, "tables", tables)
            setattr(mod, "get_table", get_table)
            setattr(mod, "get_table_norm", get_table_norm)
        except Exception:
            pass
        page_frame("Statistica", getattr(mod, "render", lambda: st.error("render() missing")))
    else:
        st.error("Modulo Statistica non disponibile.")

elif p == "probabilita":
    mod = page_modules.get("probabilita")
    if mod:
        try:
            setattr(mod, "tables", tables)
            setattr(mod, "get_table", get_table)
            setattr(mod, "get_table_norm", get_table_norm)
        except Exception:
            pass
        page_frame("Probabilità", getattr(mod, "render", lambda: st.error("render() missing")))
    else:
        st.error("Modulo Probabilità non disponibile.")

elif p == "attivita":
    mod = page_modules.get("attivita")
    if mod:
        try:
            setattr(mod, "tables", tables)
            setattr(mod, "get_table", get_table)
            setattr(mod, "get_table_norm", get_table_norm)
        except Exception:
            pass
        page_frame("Attività TLK", getattr(mod, "render", lambda: st.error("render() missing")))
    else:
        st.error("Modulo Attività non disponibile.")

elif p == "operatore":
    mod = page_modules.get("operatore")
    if mod:
        try:
            setattr(mod, "tables", tables)
            setattr(mod, "get_table", get_table)
            setattr(mod, "get_table_norm", get_table_norm)
        except Exception:
            pass
        page_frame("Scheda Operatore", getattr(mod, "render", lambda: st.error("render() missing")))
    else:
        st.error("Modulo Operatore non disponibile.")

else:
    goto("home")




