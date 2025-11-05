# eda.py
# EDA – usa tutte le righe di ogni tabella. Niente range date mostrato.
# Filtri globali (multiselect) sotto "Schema attivo": Mese, Attività, Operatore.
# In fondo: due pulsanti fucsia affiancati "Scarica HTML" e "Home".

from __future__ import annotations

from typing import List, Tuple, Optional
import time, re, io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ================== CONFIG ==================
FALLBACK_SCHEMA = "aedbdata"
ROW_LIMIT_PER_TABLE = 0  # 0 = usa tutte le righe
DARK_BG = "#0f1113"
HIST_BINS = 30
LABEL_WRAP = 16
MIN_VALID_FOR_COL = 3
MIN_DATE_VALID = 30

# alias colonne per filtri
ALIAS_DATA = ["Data","data","date","Date","timestamp","Timestamp","datetime","dt","created_at","createdAt"]
ALIAS_ATT  = ["Attivita","attivita","Attività","attività","Activity","Categoria","TipoAttivita","Tipo","Task"]
ALIAS_OP   = ["Operatore","operatore","User","Agent","Utente","OperatoreNome","Operatore_Nome"]

SCHEMA = None
DB_TABLE = None
DB_TABLE_RED = None
DB_TABLE_CALLS = None

# ================ EMBEDDED / CSV-ONLY LOADER HELPERS ================
# This module will not use SQL. It will prefer app-injected tables (get_table / tables),
# then dataset_embedded, then CSV files under ./data/.

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
    # try various common locations and separators
    p1 = Path(__file__).parent.resolve() / "data" / f"{name}.csv"
    p2 = Path.cwd() / "data" / f"{name}.csv"
    candidates = [p1, p2]
    for p in candidates:
        try:
            if p.exists():
                for sep in [';', ',', '\t', '|']:
                    try:
                        df = pd.read_csv(p, sep=sep, engine="python")
                        # if looks reasonable, return
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
    # No DB. Return fallback schema name.
    return FALLBACK_SCHEMA

@st.cache_data(ttl=60, show_spinner=False)
def _db_version(schema: str) -> str:
    # Provide a simple version string based on current timestamp to mimic vkey behavior.
    # This avoids SQL calls and still changes over time when files are updated (minute resolution).
    return time.strftime("%Y-%m-%d %H:%M:%S")

@st.cache_data(ttl=60, show_spinner=False)
def _list_tables(schema: str, vkey: str) -> pd.DataFrame:
    """
    Return available tables using (in order):
      1) app-injected 'tables' dict
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
                rows.append({"table_name": name, "table_type": "BASE TABLE", "table_rows": nrows})
            return pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
    except Exception:
        pass

    # 2) dataset_embedded
    try:
        names = _try_dataset_embedded_names()
        for n in names:
            rows.append({"table_name": n, "table_type": "BASE TABLE", "table_rows": 0})
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
def _load_table_sample(schema: str, table: str, limit: int, vkey: str) -> pd.DataFrame:
    """
    Load table data using (in order):
      1) injected get_table / get_table_norm
      2) dataset_embedded.load_table
      3) ./data/{table}.csv with common separators
    Return a DataFrame (possibly empty).
    """
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

    # 3) CSV fallback
    try:
        df = _try_csv_load(table)
        if isinstance(df, pd.DataFrame):
            return df.head(limit) if limit and limit > 0 else df
    except Exception:
        pass

    return pd.DataFrame()

# ================== UTIL ==================
def _find_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for a in aliases:
        if a in cols: return a
    lower_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower_map: return lower_map[a.lower()]
    return None

def _to_numeric_fast(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s): return s
    if s.dtype == object:
        return pd.to_numeric(
            s.astype(str).str.replace(".", "", regex=False)
                         .str.replace(",", ".", regex=False)
                         .str.replace(" ", "", regex=False),
            errors="coerce",
        )
    return pd.to_numeric(s, errors="coerce")

def _to_datetime_fast(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s): return s
    if s.dtype == object:
        sv = s.replace({"0000-00-00": None, "0000-00-00 00:00:00": None, "": None})
        return pd.to_datetime(sv, errors="coerce", utc=False)
    return pd.to_datetime(s, errors="coerce", utc=False)

def _coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = s
        elif s.dtype == object:
            ss = _to_numeric_fast(s)
            if ss.notna().sum() >= MIN_VALID_FOR_COL:
                out[c] = ss
    return pd.DataFrame(out) if out else pd.DataFrame()

def _datetime_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            if s.notna().sum() >= MIN_DATE_VALID: cols.append(c)
        elif s.dtype == object:
            s2 = _to_datetime_fast(s)
            if s2.notna().sum() >= MIN_DATE_VALID: cols.append(c)
    return cols

def _entropy(series: pd.Series, bins: int = 30) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna().values
    if s.size == 0: return np.nan
    counts, _ = np.histogram(s, bins=bins)
    p = counts.astype(float); p = p[p>0]; p = p/p.sum()
    return float(-(p*np.log2(p)).sum())

def advanced_numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        x = s.dropna().to_numpy(dtype=float)
        n, k = int(len(s)), int(x.size)
        if k == 0:
            rows.append({"colonna":c,"count":n,"valid":0,"missing_%":100.0,"unique":0,
                         "mean":np.nan,"median":np.nan,"std":np.nan,"var":np.nan,
                         "min":np.nan,"q25":np.nan,"q50":np.nan,"q75":np.nan,"max":np.nan,
                         "zeros_%":np.nan,"entropy_bits":np.nan})
            continue
        q25, q50, q75 = np.percentile(x, [25,50,75])
        rows.append({"colonna":c,"count":n,"valid":k,"missing_%":float((n-k)/max(n,1)*100),
                     "unique":int(s.nunique(dropna=True)),
                     "mean":float(np.mean(x)),"median":float(np.median(x)),
                     "std":float(np.std(x, ddof=1)) if k>1 else np.nan,
                     "var":float(np.var(x, ddof=1)) if k>1 else np.nan,
                     "min":float(np.min(x)),"q25":float(q25),"q50":float(q50),"q75":float(q75),"max":float(np.max(x)),
                     "zeros_%":float((s==0).mean()*100),"entropy_bits":np.nan})
    out = pd.DataFrame(rows)
    if not out.empty:
        order = ["colonna","count","valid","missing_%","unique","mean","median","std","var","min","q25","q50","q75","max","zeros_%","entropy_bits"]
        out = out[order].sort_values("colonna").reset_index(drop=True)
    return out

def _wrap_label(s: str, width: int = LABEL_WRAP) -> str:
    s = str(s).strip()
    if len(s)<=width: return s
    parts, line = [], ""
    for t in re.split(r"([_\-/\.\|\s])", s):
        if len(line+t)>width and line:
            parts.append(line.rstrip()); line = t.lstrip()
        else:
            line += t
    if line: parts.append(line)
    return "<br>".join(parts)

def corr_heatmap_full(df_in: pd.DataFrame) -> go.Figure:
    if df_in is None or df_in.empty: return go.Figure()
    num = _coerce_numeric_like(df_in)
    if num.empty or num.shape[1]==0: return go.Figure()
    keep = []
    for c in num.columns:
        s = pd.to_numeric(num[c], errors="coerce")
        if s.notna().sum() >= MIN_VALID_FOR_COL:
            keep.append(c)
    num = num[keep]
    if num.shape[1]==0: return go.Figure()
    corr = num.corr(method="pearson", min_periods=MIN_VALID_FOR_COL)
    xs = list(corr.columns); ys = list(corr.index)
    xt = [_wrap_label(x, LABEL_WRAP) for x in xs]
    yt = [_wrap_label(y, LABEL_WRAP) for y in ys]
    hm = go.Heatmap(
        z=corr.values, x=xs, y=ys, zmin=-1, zmax=1,
        colorscale=[[0.0,"#B22222"],[0.35,"#FF8C00"],[0.5,"#FFD700"],[1.0,"#1E7E34"]],
        xgap=2, ygap=2,
        colorbar=dict(tickvals=[-1,-0.5,0,0.5,1], len=0.6, thickness=14),
        hovertemplate="x: %{x}<br>y: %{y}<br>r: %{z:.3f}<extra></extra>"
    )
    fig = go.Figure([hm])
    fig.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                      height=max(560, len(corr)*26), margin=dict(t=60,l=120,r=120,b=120))
    fig.update_xaxes(ticktext=xt, tickvals=xs)
    fig.update_yaxes(ticktext=yt, tickvals=ys, autorange="reversed")
    return fig

def _kde_numpy(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    vals = values[~np.isnan(values)]; n = vals.size
    if n==0: return np.zeros_like(grid)
    std = np.std(vals, ddof=1) if n>1 else 0.0
    h = 1.06 * std * n ** (-1/5) if std>0 else 0.1
    if h<=0: h=0.1
    diff = (grid[:,None] - vals[None,:]) / h
    kern = np.exp(-0.5 * diff**2) / np.sqrt(2*np.pi)
    return kern.sum(axis=1) / (n*h)

def distribution_figure(series: pd.Series, name: str) -> go.Figure:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return go.Figure()
    x = s.to_numpy(dtype=float); mn, mx = float(np.min(x)), float(np.max(x))
    if np.isclose(mn, mx): mn, mx = mn-0.5, mx+0.5
    grid = np.linspace(mn, mx, 400); kde = _kde_numpy(x, grid)
    counts, edges = np.histogram(x, bins=HIST_BINS)
    centers = (edges[:-1]+edges[1:])/2
    widths = (edges[1:] - edges[:-1])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=centers, y=counts, width=widths, name="Istogramma", marker=dict(line=dict(width=0))))
    y2 = kde/kde.max()*counts.max() if kde.max()>0 else kde
    fig.add_trace(go.Scatter(x=grid, y=y2, mode="lines", name="KDE", line=dict(width=2)))
    fig.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                      height=360, legend=dict(orientation="h"),
                      margin=dict(t=40,l=60,r=40,b=50))
    fig.update_xaxes(title=name); fig.update_yaxes(title="Conteggio")
    return fig

# ================== GLOBAL FILTER OPTIONS ==================
@st.cache_data(ttl=60, show_spinner=False)
def _collect_global_filter_options(schema_now: str, vkey: str):
    mesi_set, att_set, op_set = set(), set(), set()
    tbls = _list_tables(schema_now, vkey)
    for _, r in tbls.iterrows():
        tname = r["table_name"]
        df = _load_table_sample(schema_now, tname, ROW_LIMIT_PER_TABLE, vkey)
        if df.empty: continue
        dcol = _find_col(df, ALIAS_DATA)
        if dcol:
            dt = _to_datetime_fast(df[dcol])
            if dt.notna().any():
                mesi_set.update(dt.dt.to_period("M").astype(str).dropna().unique().tolist())
        acol = _find_col(df, ALIAS_ATT)
        if acol:
            att_set.update(df[acol].dropna().astype(str).unique().tolist())
        ocol = _find_col(df, ALIAS_OP)
        if ocol:
            op_set.update(df[ocol].dropna().astype(str).unique().tolist())
    return sorted(mesi_set), sorted(att_set), sorted(op_set)

# ================== UI HELPERS ==================
def _auto_height(df: pd.DataFrame, row_px=28, base_px=60, max_px=700) -> int:
    n = len(df) if df is not None else 0
    return min(base_px + row_px * max(1, min(n, 250)), max_px)

def show_df(df: pd.DataFrame, height: int | None = None):
    _df = df if df is not None else pd.DataFrame()
    st.dataframe(_df, use_container_width=True, height=height or _auto_height(_df))

def go_home():
    st.session_state["page"] = "home"
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

def _fig_to_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")

# ================== REPORT (HTML) ==================
def _build_printable_report(schema_now: str, vkey: str) -> str:
    head = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>EDA Report</title>
<link rel="preconnect" href="https://cdn.plot.ly"/>
<style>
  body {{ font-family: Arial, sans-serif; color:#111; }}
  h1,h2,h3 {{ margin: 8px 0; }}
  .section {{ page-break-inside: avoid; margin: 18px 0; }}
  table {{ border-collapse: collapse; width:100%; font-size:12px; }}
  th,td {{ border:1px solid #ddd; padding:6px; }}
  .muted {{ color:#444; }}
</style>
</head>
<body>
<h1>EDA Report</h1>
<p class="muted">Schema: <b>{schema_now}</b> | vkey: {vkey}</p>
"""
    body = []

    tbls = _list_tables(schema_now, vkey)
    body.append("<div class='section'><h2>Tabelle nello schema</h2>")
    body.append(tbls[["table_name","table_type","table_rows"]].to_html(index=False))
    body.append("</div>")

    for _, r in tbls.iterrows():
        tname = r["table_name"]
        df = _load_table_sample(schema_now, tname, ROW_LIMIT_PER_TABLE, vkey)

        body.append(f"<div class='section'><h2>Tabella: {schema_now}.{tname}</h2>")
        body.append(f"<p class='muted'>Righe: {len(df)} | Colonne: {len(df.columns)}</p>")

        if df.empty:
            body.append("<p>Tabella vuota o non leggibile.</p></div>")
            continue

        body.append("<h3>Anteprima</h3>")
        body.append(df.head(200).to_html(index=False))

        miss = pd.DataFrame({
            "colonna": df.columns,
            "missing": [int(df[c].isna().sum()) for c in df.columns],
            "missing_%": [float(df[c].isna().mean()*100) for c in df.columns],
            "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }).sort_values("missing_%", ascending=False).reset_index(drop=True)
        body.append("<h3>Missing per colonna</h3>")
        body.append(miss.to_html(index=False))

        adv = advanced_numeric_profile(df)
        if not adv.empty:
            body.append("<h3>Statistiche avanzate (numeriche)</h3>")
            body.append(adv.to_html(index=False))

        figc = corr_heatmap_full(df)
        if len(figc.data) > 0:
            body.append("<h3>Matrice di correlazione (Pearson)</h3>")
            body.append(_fig_to_html(figc))

        num_all = _coerce_numeric_like(df)
        if not num_all.empty:
            body.append("<h3>Distribuzioni</h3>")
            for col in list(num_all.columns):
                figd = distribution_figure(num_all[col], col)
                if len(figd.data) > 0:
                    body.append(f"<h4>{col}</h4>")
                    body.append(_fig_to_html(figd))

        dcols = _datetime_columns(df)
        if dcols:
            dcol = dcols[0]
            dt = _to_datetime_fast(df[dcol]); mask = dt.notna()
            if mask.sum() >= MIN_DATE_VALID:
                tmp = df.loc[mask].copy()
                tmp["_wd"] = dt[mask].dt.weekday
                tmp["_lbl"] = tmp["_wd"].map({0:"LUN",1:"MAR",2:"MER",3:"GIO",4:"VEN",5:"SAB",6:"DOM"})
                counts = (tmp["_lbl"].value_counts()
                          .reindex(["LUN","MAR","MER","GIO","VEN","SAB","DOM"], fill_value=0)
                          .rename_axis("Giorno").reset_index(name="Conteggio"))
                body.append("<h3>Pattern giornalieri</h3>")
                body.append(counts.to_html(index=False))
                if not num_all.empty:
                    means = (pd.concat([tmp[["_lbl"]], num_all.loc[tmp.index]], axis=1)
                             .groupby("_lbl").mean(numeric_only=True)
                             .reindex(["LUN","MAR","MER","GIO","VEN","SAB","DOM"]))
                    body.append(means.rename_axis("Giorno").reset_index().to_html(index=False))
        body.append("</div>")

    tail = "</body></html>"
    return head + "\n".join(body) + tail

# ================== PAGE ==================
def render() -> None:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=60_000, key="eda_autorefresh")
    except Exception:
        pass

    schema_now = _current_db()
    vkey = f"{_db_version(schema_now)}|{int(time.time()//60)}"

    st.title("EDA - Explorative Data Analysis")
    st.caption(f"Schema attivo: {schema_now}  |  vkey: {vkey}")

    mesi_opts, att_opts, op_opts = _collect_global_filter_options(schema_now, vkey)

    c1, c2, c3 = st.columns(3)
    with c1:
        mesi_sel = st.multiselect("Mese (YYYY-MM)", mesi_opts, default=[], key="f_mesi")
    with c2:
        att_sel = st.multiselect("Attività", att_opts, default=[], key="f_att")
    with c3:
        op_sel = st.multiselect("Operatore", op_opts, default=[], key="f_op")

    tbls = _list_tables(schema_now, vkey)
    st.caption(f"Tabelle nello schema `{schema_now}`: {len(tbls)}")
    show_df(tbls[["table_name","table_type","table_rows"]], height=300)

    for _, r in tbls.iterrows():
        tname = r["table_name"]
        with st.expander(f"Tabella: {schema_now}.{tname}", expanded=False):
            df = _load_table_sample(schema_now, tname, ROW_LIMIT_PER_TABLE, vkey)
            if df.empty:
                st.warning("Tabella vuota o non leggibile.")
                continue

            dcol = _find_col(df, ALIAS_DATA)
            acol = _find_col(df, ALIAS_ATT)
            ocol = _find_col(df, ALIAS_OP)

            mask = pd.Series(True, index=df.index)

            if mesi_sel and dcol:
                dt = _to_datetime_fast(df[dcol])
                if dt.notna().any():
                    months = dt.dt.to_period("M").astype(str)
                    mask &= months.isin(mesi_sel)

            if att_sel and acol:
                mask &= df[acol].astype(str).isin(att_sel)

            if op_sel and ocol:
                mask &= df[ocol].astype(str).isin(op_sel)

            df = df.loc[mask].copy()

            rows_cnt, cols_cnt = len(df), len(df.columns)
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            dt_cols = _datetime_columns(df)
            st.caption(
                f"Righe: {rows_cnt:,} | Colonne: {cols_cnt} | Numeriche: {len(num_cols)} | Date: {len(dt_cols)}"
                .replace(",", ".")
            )

            st.subheader("Anteprima"); show_df(df.head(200), height=360)

            st.subheader("Missing per colonna")
            miss = pd.DataFrame({
                "colonna": df.columns,
                "missing": [int(df[c].isna().sum()) for c in df.columns],
                "missing_%": [float(df[c].isna().mean()*100) for c in df.columns],
                "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
            }).sort_values("missing_%", ascending=False).reset_index(drop=True)
            show_df(miss)

            st.subheader("Statistiche avanzate (numeriche)")
            adv = advanced_numeric_profile(df)
            if adv.empty: st.info("Nessuna colonna numerica nativa.")
            else: show_df(adv)

            st.subheader("Frequenze categoriali top-10")
            cats = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])]
            if cats:
                for c in cats:
                    with st.expander(c, expanded=False):
                        s = df[c].astype(str)
                        top = s.value_counts(dropna=False).head(10).reset_index()
                        top.columns = [c, "count"]; top["pct"] = top["count"]/max(len(s),1)*100
                        show_df(top)
            else:
                st.caption("Nessuna categorica testuale rilevata.")

            st.subheader("Matrice di correlazione completa")
            figc = corr_heatmap_full(df)
            if len(figc.data)==0: st.caption("Correlazioni: non applicabile.")
            else: st.plotly_chart(figc, use_container_width=True, config={"displaylogo": False})

            num_all = _coerce_numeric_like(df)
            if not num_all.empty:
                st.subheader("Distribuzioni (Istogramma + KDE)")
                for col in list(num_all.columns):
                    with st.expander(f"Distribuzione: {col}", expanded=False):
                        figd = distribution_figure(num_all[col], col)
                        if len(figd.data)==0: st.caption("Nessun dato valido.")
                        else: st.plotly_chart(figd, use_container_width=True, config={"displaylogo": False})

            if dcol:
                st.subheader("Pattern giornalieri")
                dt = _to_datetime_fast(df[dcol]); mask_dt = dt.notna()
                if mask_dt.sum() >= MIN_DATE_VALID:
                    tmp = df.loc[mask_dt].copy()
                    tmp["_wd"] = dt[mask_dt].dt.weekday
                    tmp["_lbl"] = tmp["_wd"].map({0:"LUN",1:"MAR",2:"MER",3:"GIO",4:"VEN",5:"SAB",6:"DOM"})
                    counts = (tmp["_lbl"].value_counts()
                              .reindex(["LUN","MAR","MER","GIO","VEN","SAB","DOM"], fill_value=0)
                              .rename_axis("Giorno").reset_index(name="Conteggio"))
                    show_df(counts)
                    if not num_all.empty:
                        means = (pd.concat([tmp[["_lbl"]], num_all.loc[tmp.index]], axis=1)
                                 .groupby("_lbl").mean(numeric_only=True)
                                 .reindex(["LUN","MAR","MER","GIO","VEN","SAB","DOM"]))
                        show_df(means.rename_axis("Giorno").reset_index())
                else:
                    st.caption("Date insufficienti per analisi.")
            else:
                st.caption("Nessuna colonna data rilevata.")

    st.markdown("---")

    schema_now = _current_db()
    vkey_now = f"{_db_version(schema_now)}|{int(time.time()//60)}"
    html_report = _build_printable_report(schema_now, vkey_now)
    buf = io.BytesIO(html_report.encode("utf-8"))

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

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Scarica HTML",
            data=buf,
            file_name=f"EDA_report_{schema_now}_{int(time.time())}.html",
            mime="text/html",
            key="scarica_html",
            use_container_width=True,
        )
    with c2:
        if st.button("🏠 Home", key="home_btn", use_container_width=True):
            go_home()

if __name__ == "__main__":
    render()


