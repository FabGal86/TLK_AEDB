# atts.py — Attività TLK: confronto attività + export HTML con grafici
from __future__ import annotations
from typing import List, Optional, Dict, Tuple
import io, time, re, unicodedata
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ================== COSTANTI ==================
FALLBACK_SCHEMA = "aedbdata"
ROW_LIMIT_PER_TABLE = 0
DARK_BG = "#0f1113"
FUXIA  = "#ff00ff"
GREEN  = "#00ff66"
ORANGE = "#ffa500"
CADUTI = "#4b5563"
BLUE   = "#3b82f6"
TEAL   = "#14b8a6"
YELLOW = "#f97316"
PINK   = "#e11d48"
CYAN   = "#22d3ee"
VIOLET = "#a78bfa"
COMP_COLORS = [BLUE, TEAL, YELLOW, PINK, CYAN, VIOLET, "#ef4444", "#10b981", "#64748b"]

# ================== ALIAS ==================
def _norm(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.replace(".", " ").replace("%", " ").replace("/", " ")
    s = s.replace("(", " ").replace(")", " ").replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", s).strip().lower()

ALIAS_DATA  = [_norm(x) for x in ["Data","date","timestamp","datetime","dt","created_at","createdAt"]]
ALIAS_ATT   = [_norm(x) for x in ["Attivita","attività","Activity","Categoria","TipoAttivita","Tipo","Task"]]
ALIAS_OP    = [_norm(x) for x in ["Operatore","User","Agent","Utente","OperatoreNome","Operatore_Nome"]]
ALIAS_LAV   = [_norm(x) for x in [
    "Lavorazione generale","Lavorazione  generale","lavorazionegenerale",
    "tot_lavorazione","tot_lavorazioni","tempo_lavorazione","tempo_lavorazione_generale",
    "lavorazione_totale","lav_generale","lav_generali"
]]
ALIAS_CONV  = [_norm(x) for x in [
    "Conversazione","conversazioni","ore_conversazione","h_conversazione",
    "durata_conversazione","conversazioni_ore","tempo_conversazione","talk_time"
]]
ALIAS_INCH  = [_norm(x) for x in ["In chiamata","in_chiamata","tempo_in_chiamata","tempo_chiamata","durata_chiamata"]]
ALIAS_PROC  = [_norm(x) for x in ["Processati","processato","record_processati","elaborati","processed","calls_processed"]]
ALIAS_CALLS = [_norm(x) for x in [
    "Nr. chiamate effettuate","nr chiamate effettuate","numero_chiamate",
    "tot_chiamate","chiamate","calls","call_count","n_chiamate"
]]
ALIAS_RISP  = [_norm(x) for x in [
    "Nr. chiamate con risposta","nr chiamate con risposta","risposte",
    "answered","answered_calls","responses"
]]
ALIAS_POS   = [_norm(x) for x in ["Positivi","esiti_positivi","lead_positivi","ok","esito_positivo","positivi_tot"]]
ALIAS_POSC  = [_norm(x) for x in [
    "Positivi confermati","positivi_confermati","confermati","ok_confermati",
    "positivi_ok","positivi_confirmed","confirmed"
]]

# ================== DB-FREE LOADER (like EDA) ==================
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
        gl_tables = globals().get("tables", None)
        if isinstance(gl_tables, dict) and name in gl_tables:
            df = gl_tables[name]
            if isinstance(df, pd.DataFrame):
                return df
    except Exception:
        pass
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
    p1 = Path(__file__).parent.resolve() / "data" / f"{name}.csv"
    p2 = Path.cwd() / "data" / f"{name}.csv"
    candidates = [p1, p2]
    for p in candidates:
        try:
            if p.exists():
                for sep in [';', ',', '\t', '|']:
                    try:
                        df = pd.read_csv(p, sep=sep, engine="python")
                        if df.shape[1] > 0 or df.shape[0] > 0:
                            return df
                    except Exception:
                        continue
        except Exception:
            continue
    return None

@st.cache_data(ttl=60, show_spinner=False)
def _current_db() -> str:
    # No DB. Return fallback schema name.
    return FALLBACK_SCHEMA

@st.cache_data(ttl=60, show_spinner=False)
def _db_version(schema: str) -> str:
    # Provide a simple version string based on current timestamp to mimic vkey behavior.
    return time.strftime("%Y-%m-%d %H:%M:%S")

@st.cache_data(ttl=60, show_spinner=False)
def _list_tables(schema: str) -> pd.DataFrame:
    """
    Return available tables using (in order):
      1) injected 'tables' dict
      2) dataset_embedded module
      3) CSV files in ./data
    """
    rows = []
    # 1) injected
    try:
        gl_tables = globals().get("tables", None)
        if isinstance(gl_tables, dict) and gl_tables:
            for name, df in gl_tables.items():
                try:
                    nrows = int(df.shape[0]) if hasattr(df, "shape") else 0
                except Exception:
                    nrows = 0
                rows.append({"table_name": name, "table_type": "INJECTED", "table_rows": nrows})
            return pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
    except Exception:
        pass

    # 2) dataset_embedded
    try:
        names = _try_dataset_embedded_names()
        for n in names:
            rows.append({"table_name": n, "table_type": "EMBEDDED", "table_rows": 0})
        if rows:
            return pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
    except Exception:
        pass

    # 3) CSV in ./data
    try:
        data_dir = Path(__file__).parent.resolve() / "data"
        if data_dir.exists():
            for p in sorted(data_dir.glob("*.csv")):
                rows.append({"table_name": p.stem, "table_type": "CSV", "table_rows": 0})
            if rows:
                return pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
    except Exception:
        pass

    # fallback empty frame
    return pd.DataFrame(columns=["table_name", "table_type", "table_rows"])


@st.cache_data(ttl=60, show_spinner=False)
def _load_table(schema: str, table: str, limit: int = 0) -> pd.DataFrame:
    """
    Load table data using (in order):
      1) injected get_table / get_table_norm
      2) dataset_embedded.load_table
      3) ./data/{table}.csv with common separators
    Return a DataFrame (possibly empty).
    """
    lim = int(limit) if limit and limit > 0 else None

    # 1) injected
    try:
        df = _try_get_injected_table(table)
        if isinstance(df, pd.DataFrame):
            return df.head(lim) if lim else df
    except Exception:
        pass

    # 2) dataset_embedded
    try:
        df = _try_dataset_embedded_load(table)
        if isinstance(df, pd.DataFrame):
            return df.head(lim) if lim else df
    except Exception:
        pass

    # 3) CSV fallback
    try:
        df = _try_csv_load(table)
        if isinstance(df, pd.DataFrame):
            return df.head(lim) if lim else df
    except Exception:
        pass

    return pd.DataFrame()

# ================== UTILS ==================
def _norm_cols(df: pd.DataFrame) -> dict:
    return {_norm(c): c for c in df.columns}

def _find_col(df: pd.DataFrame, aliases_norm: List[str]) -> Optional[str]:
    if df is None or df.empty: return None
    cmap = _norm_cols(df)
    for a in aliases_norm:
        if a in cmap: return cmap[a]
    for a in aliases_norm:
        for k, orig in cmap.items():
            if a in k: return orig
    return None

def _to_datetime_fast(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s): return s
    if s.dtype == object:
        sv = s.replace({"0000-00-00": None, "0000-00-00 00:00:00": None, "": None})
        return pd.to_datetime(sv, errors="coerce", utc=False, dayfirst=True, infer_datetime_format=True)
    return pd.to_datetime(s, errors="coerce", utc=False, dayfirst=True, infer_datetime_format=True)

def _to_numeric_fast(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s): return s.astype(float)
    return pd.to_numeric(
        s.astype(str).str.replace(".", "", regex=False)
                     .str.replace(",", ".", regex=False)
                     .str.replace(" ", "", regex=False),
        errors="coerce",
    ).astype(float)

def _to_hours(s: pd.Series) -> pd.Series:
    if s.dtype == object and s.astype(str).str.contains(":").any():
        td = pd.to_timedelta(s.astype(str), errors="coerce")
        return td.dt.total_seconds() / 3600.0
    num = _to_numeric_fast(s)
    vals = pd.Series(num).dropna()
    if not vals.empty and (vals.mod(1) == 0).mean() > 0.95 and (vals.median() > 59):
        return num / 60.0
    return num

def show_df(df: pd.DataFrame, height: int | None = None):
    _df = df if df is not None else pd.DataFrame()
    n = len(_df)
    h = min(60 + 28 * max(1, min(n, 250)), 700)
    st.dataframe(_df, use_container_width=True, height=height or h)

def _chart_key(attivita: str, metric: str, postfix: str = "") -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(attivita).lower()).strip("-")
    if postfix:
        slug = f"{slug}-{postfix}"
    return f"atts-{slug}-{metric}"

def _is_valid_month(m: str) -> bool:
    try:
        pd.Period(str(m), freq="M"); return True
    except Exception:
        return False

# ================== FILTRI ==================
@st.cache_data(ttl=60, show_spinner=False)
def _collect_filter_options(schema_now: str):
    mesi, att, op = set(), set(), set()
    for _, r in _list_tables(schema_now).iterrows():
        df = _load_table(schema_now, r["table_name"], ROW_LIMIT_PER_TABLE)
        if df.empty: continue
        dcol = _find_col(df, ALIAS_DATA)
        if dcol:
            dt = _to_datetime_fast(df[dcol])
            if dt.notna().any():
                mesi.update(dt.dt.to_period("M").astype(str).unique().tolist())
        acol = _find_col(df, ALIAS_ATT)
        if acol: att.update(df[acol].dropna().astype(str).unique().tolist())
        ocol = _find_col(df, ALIAS_OP)
        if ocol: op.update(df[ocol].dropna().astype(str).unique().tolist())
    return sorted([m for m in mesi if _is_valid_month(m)]), sorted(att), sorted(op)

# ================== AGGREGAZIONE MENSILE ==================
@st.cache_data(ttl=60, show_spinner=False)
def _monthly_by_activity(schema_now: str,
                         mesi_sel: List[str], op_sel: List[str]) -> pd.DataFrame:
    rows = []
    for _, r in _list_tables(schema_now).iterrows():
        df = _load_table(schema_now, r["table_name"], ROW_LIMIT_PER_TABLE)
        if df.empty: continue

        dcol = _find_col(df, ALIAS_DATA)
        acol = _find_col(df, ALIAS_ATT)
        ocol = _find_col(df, ALIAS_OP)
        if not (dcol and acol and ocol): continue

        lav = _find_col(df, ALIAS_LAV)
        conv = _find_col(df, ALIAS_CONV) or _find_col(df, ALIAS_INCH)
        proc = _find_col(df, ALIAS_PROC)
        calls= _find_col(df, ALIAS_CALLS)
        risp = _find_col(df, ALIAS_RISP)
        pos  = _find_col(df, ALIAS_POS)
        posc = _find_col(df, ALIAS_POSC)
        if not any([lav, conv, proc, calls, risp, pos, posc]): continue

        dt = _to_datetime_fast(df[dcol])
        month = dt.dt.to_period("M").astype(str)

        mask = pd.Series(True, index=df.index)
        if mesi_sel:
            valid_mesi = [m for m in mesi_sel if _is_valid_month(m)]
            mask &= month.isin(valid_mesi)
        if op_sel:  mask &= df[ocol].astype(str).isin(op_sel)
        if not mask.any(): continue

        tmp = pd.DataFrame({
            "mese": month[mask],
            "attivita": df.loc[mask, acol].astype(str),
            "lav_gen": _to_hours(df.loc[mask, lav]) if lav else 0.0,
            "conversazione": _to_hours(df.loc[mask, conv]) if conv else 0.0,
            "processati": _to_numeric_fast(df.loc[mask, proc]) if proc else 0.0,
            "chiamate": _to_numeric_fast(df.loc[mask, calls]) if calls else 0.0,
            "risposte": _to_numeric_fast(df.loc[mask, risp]) if risp else 0.0,
            "positivi": _to_numeric_fast(df.loc[mask, pos]) if pos else 0.0,
            "positivi_conf": _to_numeric_fast(df.loc[mask, posc]) if posc else 0.0,
        })
        tmp = tmp[tmp["mese"].apply(_is_valid_month)]
        if not tmp.empty:
            rows.append(tmp)

    if not rows:
        cols = ["mese","attivita","lav_gen","conversazione","processati","chiamate","risposte","positivi","positivi_conf",
                "proc_per_pos","resa_h","caduti_pct","red_pct","risposta_pct","in_ch_pct"]
        return pd.DataFrame(columns=cols)

    df_all = pd.concat(rows, ignore_index=True)

    agg = (df_all.groupby(["mese","attivita"], as_index=False)
           .sum(numeric_only=True))

    # assicurati tipi numerici e gestione inf/nan
    for c in ["positivi","processati","lav_gen","conversazione","chiamate","risposte","positivi_conf"]:
        if c in agg.columns:
            agg[c] = pd.to_numeric(agg[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pos   = agg["positivi"].to_numpy(float)
    proc  = agg["processati"].to_numpy(float)
    lav   = agg["lav_gen"].to_numpy(float)
    calls = agg["chiamate"].to_numpy(float)

    agg["proc_per_pos"] = np.where(pos > 0, proc / pos, 0.0)
    agg["resa_h"]       = np.where(lav > 0,  agg["positivi"] / lav, 0.0)
    agg["caduti_pct"]   = np.where(pos > 0, (agg["positivi"] - agg["positivi_conf"]) / pos * 100.0, 0.0)
    agg["red_pct"]      = np.where(proc > 0, agg["positivi"] / proc * 100.0, 0.0)
    agg["risposta_pct"] = np.where(calls > 0, agg["risposte"] / calls * 100.0, 0.0)
    agg["in_ch_pct"]    = np.where(lav > 0,  agg["conversazione"] / lav * 100.0, 0.0)

    # clamp percentuali tra 0 e 100 per robustezza
    for pct in ["caduti_pct","red_pct","risposta_pct","in_ch_pct"]:
        if pct in agg.columns:
            agg[pct] = pd.to_numeric(agg[pct], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            agg[pct] = agg[pct].clip(lower=0.0, upper=100.0)

    try:
        agg["_ord"] = pd.to_datetime(agg["mese"] + "-01", errors="coerce")
        agg = agg.sort_values(["attivita","_ord"]).drop(columns="_ord")
    except Exception:
        agg = agg.sort_values(["attivita","mese"])

    return agg.reset_index(drop=True)

# ================== SERIE MENSILI ==================
def _series_monthly(dfm: pd.DataFrame, att: Optional[str], metric: str) -> pd.Series:
    if dfm.empty: return pd.Series(dtype=float)
    if att is None:
        s = dfm.groupby("mese")[metric].sum(numeric_only=True)
    else:
        s = dfm.loc[dfm["attivita"] == att].groupby("mese")[metric].sum(numeric_only=True)
    s = s[pd.to_datetime(s.index + "-01", errors="coerce").notna()]
    s.index = pd.to_datetime(s.index + "-01").to_period("M").astype(str)
    return s

def _series_monthly_mean(dfm: pd.DataFrame, metric: str) -> pd.Series:
    if dfm.empty: return pd.Series(dtype=float)
    s = dfm.groupby("mese")[metric].mean(numeric_only=True)
    s = s[pd.to_datetime(s.index + "-01", errors="coerce").notna()]
    s.index = pd.to_datetime(s.index + "-01").to_period("M").astype(str)
    return s

# ================== GRAFICO: BARRE + CONFRONTO ==================
def _bar_with_trend_compare(
    base: pd.Series,
    comps: List[Tuple[str, pd.Series, str]],
    title: str,
    y_title: str,
    base_color: str,
    base_label: str,
    trend_color: str = ORANGE,
) -> go.Figure:
    def _norm_s(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        s = s.copy()
        s.index = pd.to_datetime(pd.Index(s.index).astype(str) + "-01", errors="coerce") \
                    .to_period("M").astype(str)
        s = s.groupby(level=0).sum()
        return s

    base = _norm_s(base)
    comps = [(name, _norm_s(s), col) for name, s, col in comps]

    months = sorted(set(base.index.tolist() + [m for _, s, _ in comps for m in s.index.tolist()]))
    fig = go.Figure()
    if months:
        fig.add_bar(name=base_label, x=months, y=[float(base.get(m, 0.0)) for m in months],
                    marker_color=base_color, showlegend=True)
        for name, s, col in comps:
            fig.add_bar(name=name, x=months, y=[float(s.get(m, 0.0)) for m in months],
                        marker_color=col, showlegend=True)

        try:
            y_base = [float(base.get(m, 0.0)) for m in months]
            if len(months) >= 2 and not np.allclose(y_base, 0):
                xdt = pd.to_datetime([m + "-01" for m in months], errors="coerce")
                xnum = xdt.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
                a, b = np.polyfit(xnum, np.array(y_base, dtype=float), 1)
                y0, y1 = a*xnum.min() + b, a*xnum.max() + b
                fig.add_trace(go.Scatter(
                    x=[months[0], months[-1]],
                    y=[y0, y1],
                    mode="lines+markers",
                    line=dict(width=3, color=trend_color),
                    marker=dict(size=6, color=trend_color),
                    name=f"Trend {base_label}",
                    showlegend=True,
                ))
        except Exception:
            pass

    fig.update_xaxes(type="category", categoryorder="array", categoryarray=months)
    fig.update_layout(
        title=dict(text=title, pad=dict(t=6)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=420, xaxis_title="", yaxis_title=y_title,
        margin=dict(t=60, l=60, r=60, b=80),
        barmode="group",
        showlegend=True,
    )
    return fig

def _fig_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displaylogo": False})

# ================== PAGINA ==================
def render() -> None:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60_000, key="atts_autorefresh")
    except Exception:
        pass

    st.title("Attività TLK")
    schema_now = _current_db()
    vkey = f"{_db_version(schema_now)}|{int(time.time()//60)}"
    st.caption(f"Schema attivo: {schema_now}  |  vkey: {vkey}")

    mesi_opts, att_opts, op_opts = _collect_filter_options(schema_now)
    c1, c3 = st.columns(2)
    with c1:
        mesi_sel = st.multiselect("Mese (YYYY-MM)", mesi_opts, default=[], key="atts_mesi")
    with c3:
        op_sel = st.multiselect("Operatore", op_opts, default=[], key="atts_op")

    dfm_all  = _monthly_by_activity(schema_now, mesi_sel, op_sel)
    if dfm_all.empty:
        st.warning("Nessun dato mensile con i filtri correnti.")
        _footer(schema_now, mesi_sel, [], op_sel, pd.DataFrame(), [])
        return

    compare_options = ["Media attività"] + sorted(dfm_all["attivita"].dropna().unique().tolist())
    compare_sel = st.multiselect("Compara con", compare_options, default=[], key="atts_compare")

    acts = sorted(dfm_all["attivita"].dropna().unique().tolist())
    tabs = st.tabs(acts)

    COLORS = {
        "lav_gen": FUXIA, "conversazione": GREEN, "chiamate": BLUE, "processati": TEAL,
        "proc_per_pos": "#22c55e", "resa_h": "#10b981", "positivi_conf": ORANGE,
        "caduti_pct": CADUTI, "red_pct": "#0ea5e9", "risposta_pct": "#f59e0b", "in_ch_pct": YELLOW,
    }
    metrics = [
        ("lav_gen", "Lavorazione generale", "Ore"),
        ("conversazione", "Conversazione", "Ore"),
        ("chiamate", "Nr. chiamate", "Chiamate"),
        ("proc_per_pos", "Processati per positivo", "x"),
        ("resa_h", "Resa/h (positivi / lav. gen.)", "Positivi/ora"),
        ("positivi_conf", "App confermati", "Numero"),
        ("caduti_pct", "% caduti", "%"),
        ("red_pct", "RED% mensile", "%"),
        ("risposta_pct", "Risposta % (risposte/chiamate)", "%"),
        ("in_ch_pct", "In Chiamata % (conv/lav_gen)", "%"),
    ]

    export_charts: List[Tuple[str, str]] = []

    for tab, att in zip(tabs, acts):
        with tab:
            st.subheader(att)
            sub_tab = dfm_all[dfm_all["attivita"] == att].copy()
            if mesi_sel: sub_tab = sub_tab[sub_tab["mese"].isin(mesi_sel)]

            for (mcol, title, ylab) in metrics:
                base_s = _series_monthly(sub_tab, None, mcol)

                comp_series: List[Tuple[str, pd.Series, str]] = []
                color_idx = 0
                for label in compare_sel:
                    if label == "Media attività":
                        s = _series_monthly_mean(dfm_all if not mesi_sel else dfm_all[dfm_all["mese"].isin(mesi_sel)], mcol)
                        comp_series.append((label, s, COMP_COLORS[color_idx % len(COMP_COLORS)]))
                        color_idx += 1
                    else:
                        s = _series_monthly(dfm_all if not mesi_sel else dfm_all[dfm_all["mese"].isin(mesi_sel)], label, mcol)
                        comp_series.append((label, s, COMP_COLORS[color_idx % len(COMP_COLORS)]))
                        color_idx += 1

                fig = _bar_with_trend_compare(
                    base=base_s, comps=comp_series,
                    title=title, y_title=ylab, base_color=COLORS.get(mcol, FUXIA),
                    base_label=att,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False},
                                key=_chart_key(att, mcol, "cmp"))
                export_charts.append((f"{att} — {title}", _fig_html(fig)))

            st.markdown("#### Tabella mensile (1 riga/mese)")
            cols = ["mese","lav_gen","conversazione","chiamate","processati","positivi","positivi_conf",
                    "proc_per_pos","resa_h","caduti_pct","red_pct","risposta_pct","in_ch_pct"]
            tdf = (sub_tab[cols]
                   .groupby("mese", as_index=False).sum(numeric_only=True)
                   .sort_values("mese"))
            show_df(tdf)

    _footer(schema_now, mesi_sel, [], op_sel, dfm_all, export_charts)

# ================== FOOTER + EXPORT ==================
def _footer(schema_now: str, mesi_sel: List[str], att_sel: List[str], op_sel: List[str],
            dfm: pd.DataFrame, charts: List[Tuple[str,str]]):
    st.markdown("""
    <style>
      div[data-testid="stDownloadButton"] > button,
      div[data-testid="stButton"] > button {
        background: #ff00ff !important; color: #0b0b0b !important;
        border: none !important; border-radius: 18px !important;
        padding: 22px 18px !important; font-size: 20px !important; font-weight: 600 !important;
        width: 100% !important; height: 86px !important; box-shadow: none !important;
      }
    </style>
    """, unsafe_allow_html=True)
    st.divider()
    cdl, ch = st.columns(2)
    export_cols = ["attivita","mese","lav_gen","conversazione","processati","chiamate","risposte",
                   "positivi","positivi_conf","proc_per_pos","resa_h","caduti_pct","red_pct","risposta_pct","in_ch_pct"]
    dfe = dfm[export_cols] if not dfm.empty else pd.DataFrame(columns=export_cols)
    html = _export_html(schema_now, {"mesi": mesi_sel, "operatori": op_sel}, dfe, charts)
    cdl.download_button("⬇️ Scarica HTML", data=io.BytesIO(html),
                        file_name=f"attivita_tlk_{schema_now}_{int(time.time())}.html",
                        mime="text/html", use_container_width=True, key="atts-download-html")
    if ch.button("🏠 Home", use_container_width=True, key="atts-go-home"):
        st.session_state["page"] = "home"; st.rerun()

def _export_html(schema_now: str, filters: dict, df_detail: pd.DataFrame,
                 charts: List[Tuple[str,str]]) -> bytes:
    head = f"""<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>Attività TLK</title>
<style>
body {{ font-family: Arial, sans-serif; color:#111; }}
h1,h2,h3 {{ margin: 8px 0; }}
.section {{ page-break-inside: avoid; margin: 18px 0; }}
table {{ border-collapse: collapse; width:100%; font-size:12px; }}
th,td {{ border:1px solid #ddd; padding:6px; }}
.muted {{ color:#444; }}
</style></head><body>
<h1>Attività TLK</h1>
<p class="muted">Schema: <b>{schema_now}</b></p>
<p class="muted">Filtri: {filters}</p>
<div class='section'><h2>Tabella aggregata</h2>
{df_detail.to_html(index=False)}
</div>
"""
    parts = [head]
    for title, html_fig in charts:
        parts.append(f"<div class='section'><h3>{title}</h3>{html_fig}</div>")
    parts.append("</body></html>")
    return "".join(parts).encode("utf-8")

if __name__ == "__main__":
    render()
