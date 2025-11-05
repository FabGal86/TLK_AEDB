# Stats.py
from __future__ import annotations
import io, time, re, math
from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ---- prova SciPy (ANOVA, Kruskal, Chi2) ----
try:
    from scipy import stats as sp_stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ====== COSTANTI STILE ======
DARK_BG = "#0f1113"
FUXIA  = "#ff00ff"
GREEN  = "#00ff66"
ORANGE = "#ffa500"
BLUE   = "#3b82f6"
TEAL   = "#14b8a6"
YELLOW = "#f97316"
VIOLET = "#a855f7"
GREY_OVERLAY = "rgba(148,163,184,0.22)"

# ====== ALIAS CAMPI ======
ALIAS_DATA = ["Data","data","date","Date","timestamp","Timestamp","datetime","dt","created_at","createdAt"]
ALIAS_ATT  = ["Attivita","attivita","Attività","attività","Activity","Categoria","TipoAttivita","Tipo","Task"]
ALIAS_OP   = ["Operatore","operatore","User","Agent","Utente","OperatoreNome","Operatore_Nome"]

ALIAS_POS  = ["positivi","positive","esiti_positivi","lead_positivi","ok","esito_positivo","positivi_tot"]
ALIAS_POS_CONF = [
    "Positivi confermati","positivi_confermati","confermati","ok_confermati","positivi_ok",
    "esito_confermato","positiviConfermati","positivi_confermate","confirmed","positivi_confirmed"
]
ALIAS_PROC = ["processati","processato","record_processati","contatti_processati","processed","calls","call_count"]

ALIAS_CONV = [
    "conversazione","conversazioni","ore_conversazione","h_conversazione","durata_conversazione",
    "conversazioni_ore","ore","minuti_conversazione","durata_minuti","tempo_conversazione","talk_time"
]

ALIAS_LAV_GEN = [
    "lavorazione_generale","lavorazione generale","Lavorazione Generale","tot_lavorazione",
    "tot_lavorazioni","tempo_lavorazione","tempo_lavorazione_generale","lavorazionegenerale",
    "lav_generale","lav_generali","lavorazione_totale"
]
ALIAS_IN_CHIAMATA = [
    "in_chiamata","In chiamata","in chiamata","tempo_in_chiamata","tempo_chiamata","durata_chiamata"
]

# telefonate effettuate/risposte
ALIAS_CALLS_MADE = ["chiamate_effettuate","calls_made","outbound_calls","n_chiamate","tot_chiamate"]
ALIAS_CALLS_ANSWERED = ["chiamate_risposte","answered_calls","calls_answered","chiamate_ok","risposte"]

# ====== DB-FREE / EMBEDDED CSV HELPERS ======
PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"

def _try_get_injected_table(name: str) -> Optional[pd.DataFrame]:
    """
    Cerca in globals() una mappa 'tables' oppure una funzione 'get_table(name)' fornita dall'ambiente.
    Questo permette di iniettare DataFrame direttamente dall'app chiamante senza usare DB.
    """
    try:
        gl_tables = globals().get("tables", None)
        if isinstance(gl_tables, dict) and name in gl_tables:
            df = gl_tables[name]
            if isinstance(df, pd.DataFrame):
                return df.copy()
    except Exception:
        pass
    try:
        gf = globals().get("get_table", None)
        if callable(gf):
            df = gf(name)
            if isinstance(df, pd.DataFrame):
                return df.copy()
    except Exception:
        pass
    return None

def _try_dataset_embedded_names():
    """Se esiste un modulo dataset_embedded esposto dall'ambiente, usa le sue API."""
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
    """
    Cerca file CSV in ./data/<name>.csv o ./<cwd>/data/<name>.csv.
    Prova separatori comuni. Ritorna DataFrame o None.
    """
    p1 = DATA_DIR / f"{name}.csv"
    p2 = Path.cwd() / "data" / f"{name}.csv"
    candidates = [p1, p2]
    for p in candidates:
        try:
            if p.exists():
                for sep in [',', ';', '\t', '|']:
                    try:
                        df = pd.read_csv(p, sep=sep, engine="python")
                        # accetta anche DF vuoti ma con colonne
                        if df.shape[1] > 0 or df.shape[0] >= 0:
                            return df
                    except Exception:
                        continue
        except Exception:
            continue
    return None

# ====== DB HELPERS REPLACEMENTS (uso file embedded / globals) ======
def _sql_read(q: str, params: dict | None = None) -> pd.DataFrame:
    """
    Funzione placeholder mantenuta per compatibilità. Non accede a DB.
    """
    # Non eseguiamo SQL. Ritorniamo DataFrame vuoto.
    return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def _current_db() -> str:
    # Restituisce schema fittizio. Evitiamo chiamate SQL.
    return "aedbdata"

@st.cache_data(ttl=60, show_spinner=False)
def _db_version(schema: str) -> str:
    # Versione sintetica basata su timestamp per invalidare cache quando necessario.
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

@st.cache_data(ttl=60, show_spinner=False)
def _list_tables(schema: str, vkey: str) -> pd.DataFrame:
    """
    Lista tabelle disponibili. Cerca in globals()['tables'], dataset_embedded, o CSV in ./data.
    Ritorna DataFrame con colonna 'table_name'.
    """
    rows = []
    # 1) globals() tables dict
    try:
        gl_tables = globals().get("tables", None)
        if isinstance(gl_tables, dict) and gl_tables:
            for name in sorted(gl_tables.keys()):
                rows.append({"table_name": name})
            return pd.DataFrame(rows)
    except Exception:
        pass
    # 2) dataset_embedded module
    try:
        names = _try_dataset_embedded_names()
        for n in sorted(names):
            rows.append({"table_name": n})
        if rows:
            return pd.DataFrame(rows)
    except Exception:
        pass
    # 3) CSV files in ./data
    try:
        if DATA_DIR.exists():
            for p in sorted(DATA_DIR.glob("*.csv")):
                rows.append({"table_name": p.stem})
            if rows:
                return pd.DataFrame(rows)
    except Exception:
        pass
    # empty fallback
    return pd.DataFrame(columns=["table_name"])

@st.cache_data(ttl=60, show_spinner=False)
def _load_table(schema: str, table: str, limit: int, vkey: str) -> pd.DataFrame:
    """
    Carica una 'tabella' da sorgenti locali:
      - dict 'tables' in globals
      - modulo dataset_embedded (se presente)
      - file CSV in ./data/<table>.csv o ./data/<table>.CSV
    Applica 'limit' se >0.
    """
    # 1) injected tables
    try:
        df = _try_get_injected_table(table)
        if isinstance(df, pd.DataFrame):
            return df.head(limit) if limit and limit > 0 else df
    except Exception:
        pass
    # 2) dataset_embedded loader
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

# ====== UTIL ======
def _find_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    if df is None or df.empty: return None
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

# ====== RACCOLTA DATI NORMALIZZATI ======
@st.cache_data(ttl=60, show_spinner=False)
def _collect_unified(schema: str, vkey: str, row_limit: int = 0) -> pd.DataFrame:
    """
    Ritorna righe normalizzate con campi:
    date, month, operator, operator_short, activity,
    positivi, positivi_conf, processati,
    conversazione_h, lav_gen_h, in_chiamata_h,
    calls_made, calls_answered,
    red_pct, redn_pct, rate_inch_lav_pct, rate_answ_calls_pct, proc_per_pos
    """
    rows = []
    tbls = _list_tables(schema, vkey)
    for _, r in tbls.iterrows():
        t = r["table_name"]
        df = _load_table(schema, t, row_limit, vkey)
        if df.empty:
            continue

        dcol = _find_col(df, ALIAS_DATA)
        ocol = _find_col(df, ALIAS_OP)
        acol = _find_col(df, ALIAS_ATT)

        pcol = _find_col(df, ALIAS_POS)
        pccol = _find_col(df, ALIAS_POS_CONF)
        prcol = _find_col(df, ALIAS_PROC)

        conv = _find_col(df, ALIAS_CONV)
        lavg = _find_col(df, ALIAS_LAV_GEN) or _find_col_contains(df, "lavorazione")
        inch = _find_col(df, ALIAS_IN_CHIAMATA) or _find_col_contains(df, "chiamata")

        # chiamate: prima prova alias, poi contains, poi fallback per colonne J/K (indice 9/10)
        cmade = _find_col(df, ALIAS_CALLS_MADE) or _find_col_contains(df, "chiamate")
        cansw = _find_col(df, ALIAS_CALLS_ANSWERED) or _find_col_contains(df, "risposte")

        # Fallback positionale: se non riconosciute e il file ha colonne sufficienti,
        # assumiamo J = col index 9 (10a colonna) = calls_made, K = index 10 (11a colonna) = calls_answered
        try:
            ncols = df.shape[1]
            if not cmade and ncols >= 10:
                # colonna J (10th column) -> index 9
                cmade = df.columns[9]
            if not cansw and ncols >= 11:
                # colonna K (11th column) -> index 10
                cansw = df.columns[10]
        except Exception:
            pass

        if not any([pcol, pccol, prcol, conv, lavg, inch, cmade, cansw]):
            continue

        date = _to_datetime_fast(df[dcol]) if dcol else pd.to_datetime(pd.Series([None]*len(df)))
        month = date.dt.to_period("M").astype(str) if dcol else pd.Series([None]*len(df))
        op = df[ocol].astype(str) if ocol else pd.Series([""]*len(df))
        att = df[acol].astype(str) if acol else pd.Series([""]*len(df))

        rec = pd.DataFrame({
            "date": date,
            "month": month,
            "operator": op,
            "operator_short": op.map(_short_name),
            "activity": att,
            "positivi": _to_numeric_fast(df[pcol]) if pcol else 0,
            "positivi_conf": _to_numeric_fast(df[pccol]) if pccol else 0,
            "processati": _to_numeric_fast(df[prcol]) if prcol else 0,
            "conversazione_h": _to_hours(df[conv]) if conv else 0.0,
            "lav_gen_h": _to_hours(df[lavg]) if lavg else 0.0,
            "in_chiamata_h": _to_hours(df[inch]) if inch else 0.0,
            "calls_made": _to_numeric_fast(df[cmade]) if cmade else 0,
            "calls_answered": _to_numeric_fast(df[cansw]) if cansw else 0,
        })
        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=[
            "date","month","operator","operator_short","activity",
            "positivi","positivi_conf","processati",
            "conversazione_h","lav_gen_h","in_chiamata_h",
            "calls_made","calls_answered",
            "red_pct","redn_pct","rate_inch_lav_pct","rate_answ_calls_pct","proc_per_pos"
        ])

    all_df = pd.concat(rows, ignore_index=True).fillna(0)
    # derivati
    all_df["red_pct"] = np.where(all_df["processati"]>0, all_df["positivi"]/all_df["processati"]*100, np.nan)
    all_df["redn_pct"] = np.where(all_df["processati"]>0, all_df["positivi_conf"]/all_df["processati"]*100, np.nan)  # NEW
    all_df["rate_inch_lav_pct"] = np.where(all_df["lav_gen_h"]>0, all_df["in_chiamata_h"]/all_df["lav_gen_h"]*100, np.nan)
    # rate_answ_calls_pct calcolato come Col K / Col J (calls_answered / calls_made)
    all_df["rate_answ_calls_pct"] = np.where(all_df["calls_made"]>0, all_df["calls_answered"]/all_df["calls_made"]*100, np.nan)
    all_df["proc_per_pos"] = np.where(all_df["positivi"]>0, all_df["processati"]/all_df["positivi"], np.nan)
    all_df["operator_short"] = all_df["operator_short"].where(all_df["operator_short"].astype(str).str.strip()!="", all_df["operator"])
    return all_df

# ====== PLOT HELPERS ======
def _boxplot_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    y_title: str,
) -> go.Figure:
    fig = go.Figure()
    if df.empty or value_col not in df.columns or group_col not in df.columns:
        fig.update_layout(
            title=dict(text=title, pad=dict(t=6)),
            template="plotly_dark",
            paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
            height=520, yaxis_title=y_title, showlegend=False,
            margin=dict(t=70, l=60, r=60, b=120),
        )
        return fig
    groups = sorted([g for g in df[group_col].astype(str).unique().tolist() if str(g).strip() != ""])
    palette = [FUXIA, GREEN, ORANGE, BLUE, TEAL, YELLOW, VIOLET]
    for i, g in enumerate(groups):
        sub = df.loc[df[group_col].astype(str) == g, value_col].dropna()
        if sub.empty:
            continue
        fig.add_trace(go.Box(
            y=sub,
            name=str(g),
            marker=dict(color=palette[i % len(palette)]),
            boxmean="sd",
            whiskerwidth=0.9,
            showlegend=False,
        ))
    fig.update_layout(
        title=dict(text=title, pad=dict(t=6)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=520, yaxis_title=y_title, showlegend=False,
        margin=dict(t=70, l=60, r=60, b=160),
    )
    fig.update_xaxes(tickangle=-60)
    return fig

def _heatmap_from_matrix(z: np.ndarray, xlabels: List[str], ylabels: List[str], title: str, ztitle: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=xlabels, y=ylabels, coloraxis="coloraxis"
        )
    )
    fig.update_layout(
        title=dict(text=title, pad=dict(t=6)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=520, margin=dict(t=70, l=60, r=60, b=120),
        coloraxis=dict(colorscale="Viridis", colorbar=dict(title=ztitle))
    )
    return fig

# ====== CORRELAZIONI ======
def _corr_features_vs_targets(df: pd.DataFrame) -> pd.DataFrame:
    # features quantitative candidate
    feats = ["conversazione_h","lav_gen_h","in_chiamata_h","calls_made","calls_answered","rate_inch_lav_pct","rate_answ_calls_pct","proc_per_pos"]
    targs = ["positivi","positivi_conf","red_pct"]
    cols_present = [c for c in feats+targs if c in df.columns]
    if not cols_present:
        return pd.DataFrame()
    sub = df[cols_present].copy()
    # Spearman per robustezza
    corr = sub.corr(method="spearman")
    # estrai solo righe feature e colonne target
    rows = [c for c in feats if c in corr.index]
    cols = [c for c in targs if c in corr.columns]
    return corr.loc[rows, cols]

def _operator_activity_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """Ritorna tabella contingenza operator_short x activity e Cramér's V totale."""
    if df.empty:
        return pd.DataFrame(), np.nan
    tab = pd.crosstab(df["operator_short"].astype(str), df["activity"].astype(str))
    if tab.empty:
        return tab, np.nan
    if SCIPY_OK:
        chi2, p, dof, exp = sp_stats.chi2_contingency(tab.values)
        n = tab.values.sum()
        v = math.sqrt(chi2 / (n * (min(tab.shape)-1))) if min(tab.shape) > 1 and n > 0 else np.nan
    else:
        v = np.nan
    return tab, v

# ====== TEST STATISTICI ======
def _oneway_tests(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, Optional[float]]:
    """Esegue Kruskal e, se possibile, anche ANOVA. Ritorna p-values."""
    out = {"metric": value_col, "group": group_col, "kruskal_p": None, "anova_p": None, "n_groups": 0}
    if df.empty or group_col not in df.columns or value_col not in df.columns:
        return out
    series = df[[group_col, value_col]].dropna()
    if series.empty:
        return out
    groups = []
    for g, sub in series.groupby(group_col):
        vals = pd.to_numeric(sub[value_col], errors="coerce").dropna().values
        if len(vals) > 0:
            groups.append(vals)
    out["n_groups"] = len(groups)
    if len(groups) < 2:
        return out
    # Kruskal
    if SCIPY_OK:
        try:
            kw = sp_stats.kruskal(*groups)
            out["kruskal_p"] = float(kw.pvalue)
        except Exception:
            pass
        # ANOVA classica
        try:
            an = sp_stats.f_oneway(*groups)
            out["anova_p"] = float(an.pvalue)
        except Exception:
            pass
    return out

# ====== PAGINA ======
def _go_home():
    st.session_state["page"] = "home"
    try:
        st.rerun()
    except Exception:
        pass

def render() -> None:
    # titolo
    st.title("Stats")

    # schema e versione cache
    schema_now = _current_db()
    vkey = f"{_db_version(schema_now)}|stats"

    # dati unificati
    df_all = _collect_unified(schema_now, vkey, row_limit=0)

    # filtri
    mesi_opts = sorted([m for m in df_all["month"].dropna().astype(str).unique().tolist() if m]) if not df_all.empty else []
    att_opts  = sorted([a for a in df_all["activity"].dropna().astype(str).unique().tolist() if a]) if not df_all.empty else []
    op_opts   = sorted([o for o in df_all["operator"].dropna().astype(str).unique().tolist() if o]) if not df_all.empty else []

    c1, c2, c3 = st.columns(3)
    with c1:
        mesi_sel = st.multiselect("Mese (YYYY-MM)", mesi_opts, default=[])
    with c2:
        att_sel = st.multiselect("Attività", att_opts, default=[])
    with c3:
        op_sel = st.multiselect("Operatore", op_opts, default=[])

    # applica filtri
    dff = df_all.copy()
    if mesi_sel and "month" in dff.columns:
        dff = dff[dff["month"].isin(mesi_sel)]
    if att_sel and "activity" in dff.columns:
        dff = dff[dff["activity"].astype(str).isin(att_sel)]
    if op_sel and "operator" in dff.columns:
        dff = dff[dff["operator"].astype(str).isin(op_sel)]

    # ====== BOXPLOT PER OPERATORE ======
    st.markdown("### Boxplot per Operatore")
    st.plotly_chart(_boxplot_by_group(dff, "operator_short", "positivi", "Positivi per Operatore", "n°"),
                    use_container_width=True, config={"displaylogo": False})
    st.plotly_chart(_boxplot_by_group(dff, "operator_short", "red_pct", "RED% per Operatore", "%"),
                    use_container_width=True, config={"displaylogo": False})
    # NEW: REDn% = positivi_confermati/processati
    st.plotly_chart(_boxplot_by_group(dff, "operator_short", "redn_pct", "REDn% per Operatore (conf./proc.)", "%"),
                    use_container_width=True, config={"displaylogo": False})
    st.plotly_chart(_boxplot_by_group(dff, "operator_short", "calls_made", "Nr chiamate effettuate per Operatore", "n°"),
                    use_container_width=True, config={"displaylogo": False})
    st.plotly_chart(_boxplot_by_group(dff, "operator_short", "conversazione_h", "Conversazione (h) per Operatore", "ore"),
                    use_container_width=True, config={"displaylogo": False})
    st.plotly_chart(_boxplot_by_group(dff, "operator_short", "rate_inch_lav_pct", "Rate In Chiamata / Lavorazione Generale", "%"),
                    use_container_width=True, config={"displaylogo": False})
    st.plotly_chart(_boxplot_by_group(dff, "operator_short", "rate_answ_calls_pct", "Rate Chiamate risposte / effettuate", "%"),
                    use_container_width=True, config={"displaylogo": False})

    # ====== BOXPLOT PER ATTIVITÀ ======
    st.markdown("### Boxplot per Attività")
    st.plotly_chart(_boxplot_by_group(dff, "activity", "positivi", "Positivi per Attività", "n°"),
                    use_container_width=True, config={"displaylogo": False})
    st.plotly_chart(_boxplot_by_group(dff, "activity", "lav_gen_h", "Ore lavorazione generale per Attività", "ore"),
                    use_container_width=True, config={"displaylogo": False})
    st.plotly_chart(_boxplot_by_group(dff, "activity", "proc_per_pos", "Processati per positivo per Attività", "ratio"),
                    use_container_width=True, config={"displaylogo": False})
    # NEW: RED% e REDn% per attività
    st.plotly_chart(_boxplot_by_group(dff, "activity", "red_pct", "RED% per Attività", "%"),
                    use_container_width=True, config={"displaylogo": False})
    st.plotly_chart(_boxplot_by_group(dff, "activity", "redn_pct", "REDn% per Attività (conf./proc.)", "%"),
                    use_container_width=True, config={"displaylogo": False})

    # ====== CORRELAZIONE FEATURE -> TARGET ======
    st.markdown("### Correlazione tra feature e target")
    corr_ft = _corr_features_vs_targets(dff)
    if corr_ft.empty:
        st.info("Dati insufficienti per calcolare la correlazione.")
    else:
        z = corr_ft.values
        fig_corr = _heatmap_from_matrix(z, corr_ft.columns.tolist(), corr_ft.index.tolist(),
                                        "Spearman ρ: feature → target", "ρ")
        st.plotly_chart(fig_corr, use_container_width=True, config={"displaylogo": False})
        st.caption("Interpretazione rapida: valori più alti in modulo indicano relazione più forte con il target selezionato.")

    # ====== CORRELAZIONE OPERATORI × ATTIVITÀ ======
    st.markdown("### Correlazione tra operatori e attività")
    tab_oa, cramer_v = _operator_activity_matrix(dff)
    if tab_oa.empty:
        st.info("Nessuna contingenza valida operatore×attività.")
    else:
        z = tab_oa.values.astype(float)
        fig_oa = _heatmap_from_matrix(z, tab_oa.columns.tolist(), tab_oa.index.tolist(),
                                      "Frequenze Operatore × Attività", "conteggi")
        st.plotly_chart(fig_oa, use_container_width=True, config={"displaylogo": False})
        if not np.isnan(cramer_v):
            st.caption(f"Cramér’s V globale: {cramer_v:.3f} (0=nessuna associazione, 1=associazione forte).")

    # ====== TEST STATISTICI IN FONDO ======
    st.markdown("### Test statistici sui grafici")
    results: List[Dict[str, Optional[float]]] = []

    # operator: metrica -> test
    for met in ["positivi","red_pct","redn_pct","calls_made","conversazione_h","rate_inch_lav_pct","rate_answ_calls_pct"]:
        results.append(_oneway_tests(dff, "operator_short", met))

    # activity: metrica -> test
    for met in ["positivi","lav_gen_h","proc_per_pos"]:
        results.append(_oneway_tests(dff, "activity", met))

    # chi2 per operatore×attività
    chi_row = {"metric": "operator×activity", "group": "contingenza", "kruskal_p": None, "anova_p": None, "n_groups": None}
    if SCIPY_OK and not tab_oa.empty and tab_oa.shape[0] > 1 and tab_oa.shape[1] > 1:
        try:
            chi2, p, dof, exp = sp_stats.chi2_contingency(tab_oa.values)
            chi_row["chi2_p"] = float(p)
            chi_row["cramers_v"] = float(cramer_v) if not np.isnan(cramer_v) else None
        except Exception:
            chi_row["chi2_p"] = None
    else:
        chi_row["chi2_p"] = None
    results.append(chi_row)

    # mostra tabella risultati
    res_df = pd.DataFrame(results)
    res_df = res_df.rename(columns={
        "metric":"Metrica",
        "group":"Gruppo",
        "kruskal_p":"p Kruskal",
        "anova_p":"p ANOVA",
        "n_groups":"#Gruppi",
        "chi2_p":"p Chi²",
        "cramers_v":"Cramér’s V"
    })
    st.dataframe(res_df, use_container_width=True, height=420)

    # Spiegazione p-value e IC
    st.markdown(
        "- **p-value**: valori a due code dai test indicati (Kruskal–Wallis, ANOVA a una via, Chi-quadrato). "
        "Soglia di significatività predefinita α=0.05.\n"
        "- **Intervallo di confidenza (IC)**: interpretazione a **95%**. "
        "I boxplot usano baffi stile Tukey (±1.5×IQR). "
        "Per i test riportiamo il p-value; gli IC espliciti non sono mostrati ma la soglia α=0.05 implica IC al 95%."
    )

    # ====== BOTTONI FUXIA COME SDR ======
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
      div[data-testid="stDownloadButton"] > button:hover,
      div[data-testid="stButton"] > button:hover {
        background: #7A00FF !important;
        color: #ffffff !important;
      }
    </style>
    """, unsafe_allow_html=True)

    # export HTML minimale
    html_export = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <title>Stats - Export</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{{font-family:Arial,Helvetica,sans-serif;margin:0;padding:24px;background:#fff;color:#111}}
    h1{{margin:0 0 12px 0}}
    .note{{margin-top:8px;font-size:14px;color:#444}}
    table{{border-collapse:collapse;width:100%;font-size:13px}}
    th,td{{border:1px solid #ddd;padding:6px}}
  </style>
</head>
<body>
  <h1>Stats</h1>
  <p>Esporta risultati e grafici della sezione Stats.</p>
  <p class="note">Generato: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
</body>
</html>"""
    buf = io.BytesIO(html_export.encode("utf-8"))

    c1b, c2b = st.columns(2)
    with c1b:
        st.download_button(
            "⬇️ Scarica HTML",
            data=buf,
            file_name=f"Stats_export_{int(time.time())}.html",
            mime="text/html",
            key="stats_html",
            use_container_width=True,
        )
    with c2b:
        if st.button("🏠 Home", key="stats_home", use_container_width=True):
            _go_home()

if __name__ == "__main__":
    render()
