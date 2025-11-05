# app.py — Home + routing (con Attività TLK e Scheda Operatore)
# Aggiornato: data loader integrato (CSV ';' preferred / dataset.py / dataset_embedded)
# Includes automatic semicolon CSV fix and light normalization helpers.

import streamlit as st
from pathlib import Path
import importlib
import pandas as pd
import sys
import shutil
import re

# === importa le pagine (assumi che esistano) ===
import eda
import SDR
import SDRML
import SDRforecast
import Stats
import probability
import atts
import operatore  # <- nuova pagina Scheda Operatore

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
# Behavior:
# 1) prefer CSV ./data/ reading semicolon first (fix immediato)
# 2) try dataset.py (if present)
# 3) try dataset_embedded (last)
# 4) fallback Desktop path r"C:\Users\HP\Desktop\DATABASE_TLK"
PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DESKTOP_CSV_DIR = Path(r"C:\Users\HP\Desktop\DATABASE_TLK")

tables = {}  # original, fixed tables
_norm_cache = {}  # normalized copies (lazy)

def _fix_semicolon_single_column(df: pd.DataFrame) -> pd.DataFrame:
    """If df has a single column containing semicolon-separated values, split it."""
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
    """Strip BOMs, whitespace, collapse multiple spaces in headers; keep case as-is."""
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
    """Trim string cells and collapse multiple spaces; keep types otherwise."""
    for c in df.select_dtypes(include=['object']).columns:
        try:
            df[c] = df[c].astype(str).str.strip().replace({'^nan$': None, '^None$': None}, regex=True)
            df[c] = df[c].str.replace(r'\s+', ' ', regex=True)
        except Exception:
            pass
    return df

def _ensure_mese_column(df: pd.DataFrame) -> pd.DataFrame:
    """Try find a date-like column and create 'mese' in YYYY-MM format if possible."""
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
            loaded = dataset.load_all_tables_from_embedded()
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
            # try common separators
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
        # try semicolon first
        for sep in [';', ',', '\t', '|']:
            try:
                df_try = pd.read_csv(p, sep=sep, engine='python')
                # heuristic: accept if more than 1 column OR non-empty
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
    # copy desktop CSVs to project data/ for reproducibility
    try:
        for p in sorted(DESKTOP_CSV_DIR.glob("*.csv")):
            dst = DATA_DIR / p.name
            if not dst.exists():
                shutil.copy2(p, dst)
    except Exception:
        pass
else:
    loader_source = "none"

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
    # lowercase column names
    df.columns = [str(c).strip().lower() for c in df.columns]
    # trim string cells
    for c in df.select_dtypes(include=['object']).columns:
        try:
            df[c] = df[c].astype(str).str.strip().replace({'^nan$': None}, regex=True)
        except Exception:
            pass
    # ensure mese column
    if 'mese' not in df.columns:
        df = _ensure_mese_column(df)
    _norm_cache[name] = df
    return df

# Put table names into session state for UI use
st.session_state.setdefault("tables_available", list(tables.keys()))
st.session_state.setdefault("data_loader_source", loader_source)

# ================ Router =================
if "page" not in st.session_state:
    st.session_state.page = "home"

def goto(p: str):
    st.session_state.page = p
    st.rerun()

def page_frame(_title: str, body_fn):
    body_fn()

# ================= Home =================
def render_home():
    st.markdown('<div class="title-wrap"><div class="title">TLK</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="home-spacer"></div>', unsafe_allow_html=True)

    # --- debug/info row under title ---
    cols_info = st.columns([1,2,2,2])
    with cols_info[0]:
        st.markdown(f"**Schema attivo:** aedbdata")
        st.write("")  # small gap
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
    try:
        eda.tables = tables
        eda.get_table = get_table
        eda.get_table_norm = get_table_norm
    except Exception:
        pass
    page_frame("Explorative Data Analysis", eda.render)
elif p == "sdr":
    try:
        SDR.tables = tables
        SDR.get_table = get_table
        SDR.get_table_norm = get_table_norm
    except Exception:
        pass
    page_frame("SDR Team", SDR.render)
elif p == "sdrml":
    try:
        SDRML.tables = tables
        SDRML.get_table = get_table
        SDRML.get_table_norm = get_table_norm
    except Exception:
        pass
    page_frame("SDR Insight ML", SDRML.render)
elif p == "sdrforecast":
    try:
        SDRforecast.tables = tables
        SDRforecast.get_table = get_table
        SDRforecast.get_table_norm = get_table_norm
    except Exception:
        pass
    page_frame("SDR Forecast", SDRforecast.render)
elif p == "statistica":
    try:
        Stats.tables = tables
        Stats.get_table = get_table
        Stats.get_table_norm = get_table_norm
    except Exception:
        pass
    page_frame("Statistica", Stats.render)
elif p == "probabilita":
    try:
        probability.tables = tables
        probability.get_table = get_table
        probability.get_table_norm = get_table_norm
    except Exception:
        pass
    page_frame("Probabilità", probability.render)
elif p == "attivita":
    try:
        atts.tables = tables
        atts.get_table = get_table
        atts.get_table_norm = get_table_norm
    except Exception:
        pass
    page_frame("Attività TLK", atts.render)
elif p == "operatore":
    try:
        operatore.tables = tables
        operatore.get_table = get_table
        operatore.get_table_norm = get_table_norm
    except Exception:
        pass
    page_frame("Scheda Operatore", operatore.render)
else:
    goto("home")
