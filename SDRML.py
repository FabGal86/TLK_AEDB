# ======================================================
# SDRML.py — compatibile Python 3.13
# Analisi performance Team con XGBoost + PyTorch (opzionali)
# Nessun scikit-learn. Legende sotto i grafici.
# Grafici per operatore: solo pallini + legenda esterna.
# ======================================================

from __future__ import annotations
from typing import List, Optional
import io, time, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ========== IMPORT OPZIONALI ==========
try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.optim as optim  # type: ignore
    _TORCH_OK = True
except Exception:  # Torch assente su Streamlit Cloud free
    torch, nn, optim = None, None, None  # type: ignore
    _TORCH_OK = False

try:
    import xgboost as xgb  # type: ignore
    _XGB_OK = True
except Exception:
    xgb = None  # type: ignore
    _XGB_OK = False

# ================== CONFIG STREAMLIT ==================
st.set_page_config(page_title="SDR Insight ML", page_icon="📈", layout="wide")

# ================== COSTANTI ==================
FALLBACK_SCHEMA = "aedbdata"
ROW_LIMIT_PER_TABLE = 0
DARK_BG = "#0f1113"
FUXIA  = "#ff00ff"
ORANGE = "#f97316"
BLUE   = "#0ea5e9"
VIOLET = "#8b5cf6"
CADUTI = "#4b5563"

# palette per operatori
OP_COLORS = [
    "#f97316","#22c55e","#06b6d4","#a855f7","#ef4444","#0ea5e9",
    "#eab308","#14b8a6","#6366f1","#f43f5e","#2dd4bf","#a3e635",
    "#fb7185","#4ade80","#38bdf8","#facc15","#c084fc","#22d3ee",
    "#ff7ab6","#7dd3fc","#bbf7d0","#fde68a","#d8b4fe","#60a5fa"
]

# ================== ALIAS COLONNE ==================
ALIAS_DATA = ["data","date","timestamp","datetime","created_at","createdat"]
ALIAS_ATT  = ["attivita","attività","activity","categoria","tipoattivita","task","tipo"]
ALIAS_OP   = ["operatore","user","agent","utente","operatorenome","operatore_nome"]
ALIAS_CONV = ["conversazione","ore_conversazione","talk_time"]
ALIAS_POS  = ["positivi","positive","lead_positivi","esito_positivo","positivi_tot"]
ALIAS_POS_CONF = ["positivi_confermati","positivi confermati","confermati","ok_confermati","confirmed"]
ALIAS_LAV_GEN = ["lavorazione_generale","lavorazione generale","lavorazione_totale","tot_lavorazione","tempo_lavorazione","tempo_lavorazione_generale"]
ALIAS_LAV_CONTATTI = ["lavorazione_contatti","lavorazione contatti","lav_contatti","contatti_lavorati"]
ALIAS_LAV_VARIE = ["lavorazione_varie","lavorazione varie","lav_varie","varie_lavorate"]
ALIAS_IN_CHIAMATA = ["in_chiamata","in chiamata","tempo_in_chiamata","tempo_chiamata","durata_chiamata"]
ALIAS_IN_ATTESA = ["in_attesa_di_chiamata","in attesa di chiamata","attesa_chiamata","tempo_attesa","in_attesa"]
ALIAS_CALLS_MADE = ["chiamate_effettuate","chiamate effettuate","call_made","calls_made","chiamate_totali","chiamate"]
ALIAS_CALLS_ANSWERED = ["chiamate_risposte","chiamate risposte","call_answered","calls_answered","chiamate_con_risposta","risposte"]
ALIAS_PROCESSED = ["processati","record_processati","contatti_processati","processed","elaborati","gestiti"]

# ================== DB-FREE / CSV / EMBEDDED LOADER HELPERS ==================
PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"

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
                        if df.shape[1] > 0 or df.shape[0] >= 0:
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
def _list_tables(schema: str) -> pd.DataFrame:
    rows = []
    try:
        gl_tables = globals().get("tables", None)
        if isinstance(gl_tables, dict) and gl_tables:
            for name, df in gl_tables.items():
                rows.append({"table_name": name})
            return pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
    except Exception:
        pass
    try:
        names = _try_dataset_embedded_names()
        for n in names:
            rows.append({"table_name": n})
        if rows:
            return pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
    except Exception:
        pass
    try:
        if DATA_DIR.exists():
            for p in sorted(DATA_DIR.glob("*.csv")):
                rows.append({"table_name": p.stem})
            if rows:
                return pd.DataFrame(rows).sort_values("table_name").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame(columns=["table_name"])

@st.cache_data(ttl=60, show_spinner=False)
def _load_table(schema: str, table: str) -> pd.DataFrame:
    # 1) injected tables in globals
    try:
        df = _try_get_injected_table(table)
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass
    # 2) dataset_embedded module
    try:
        df = _try_dataset_embedded_load(table)
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass
    # 3) local CSV file
    try:
        df = _try_csv_load(table)
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass
    return pd.DataFrame()

# ================== UTIL ==================
def _norm(s: str) -> str:
    return re.sub(r"[\s_]+","", s.lower())

def _find_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    if df is None or df.empty: return None
    norm_map = {_norm(c): c for c in list(df.columns)}
    for a in aliases:
        na = _norm(a)
        if na in norm_map: return norm_map[na]
    for a in aliases:
        na = _norm(a)
        for nc, real in norm_map.items():
            if na in nc: return real
    return None

def _to_numeric_fast(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s): return s
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )

def _to_hours(s: pd.Series) -> pd.Series:
    if s.dtype == object and s.astype(str).str.contains(":").any():
        td = pd.to_timedelta(s.astype(str), errors="coerce")
        return td.dt.total_seconds() / 3600.0
    return _to_numeric_fast(s)

def _short_name(full: str) -> str:
    if not isinstance(full, str) or not full.strip():
        return "UNKNOWN"
    p = full.strip().split()
    if len(p) == 1:
        return p[0].title()
    surname = " ".join(p[:-1]).title()
    iniz = p[-1][0].upper()
    return f"{surname} {iniz}"

def _op_color_map(ops: List[str]) -> dict:
    cmap = {}
    for i, op in enumerate(sorted(ops)):
        cmap[op] = OP_COLORS[i % len(OP_COLORS)]
    return cmap

def _legend_ops_html(op_list: List[str], color_map: dict) -> str:
    items = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 12px;">'
        f'<span style="width:14px;height:14px;background:{color_map[o]};display:inline-block;border-radius:50%;"></span>'
        f'<span style="font-size:13px;color:#ffffff;">{o}</span></div>'
        for o in op_list
    )
    return (
        "<div style='display:flex;flex-wrap:wrap;align-items:center;justify-content:center;"
        "margin:8px 0 16px 0;'>"
        f"{items}</div>"
    )

# ================== MODELLI: TORCH/XGB CON FALLBACK NUMPY ==================
class _RidgeReg:
    """Ridge lineare puro NumPy come fallback a XGBoost."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, float); y = np.asarray(y, float)
        n, d = X.shape
        A = X.T @ X + self.alpha * np.eye(d)
        b = X.T @ y
        self.coef_ = np.linalg.solve(A, b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, float) @ (self.coef_ if self.coef_ is not None else 0.0)

    @property
    def feature_importances_(self) -> np.ndarray:
        c = np.abs(self.coef_ if self.coef_ is not None else np.zeros(1))
        s = c.sum()
        return c / s if s > 0 else c

# Torch regressor semplice
class LinearNet(nn.Module if _TORCH_OK else object):
    def __init__(self, n_in: int, n_out: int = 1):
        if not _TORCH_OK: return
        super().__init__()  # type: ignore
        self.net = nn.Sequential(
            nn.Linear(n_in, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_out)
        )
    def forward(self, x):  # type: ignore
        return self.net(x)

def train_torch_regressor(X: np.ndarray, y: np.ndarray, epochs: int = 250, lr: float = 0.01):
    if not _TORCH_OK:
        return None
    model = LinearNet(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()
    model.eval()
    return model

def train_xgb_regressor(X: np.ndarray, y: np.ndarray):
    if _XGB_OK:
        m = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="reg:squarederror", n_jobs=4, verbosity=0
        )
        m.fit(X, y)
        return m
    # Fallback: Ridge
    return _RidgeReg(alpha=1.0).fit(X, y)

def predict_torch(model, X: np.ndarray) -> np.ndarray:
    if not _TORCH_OK or model is None:
        return np.zeros(len(X))
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy().ravel()

def detect_anomalies_xgb(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if df.empty or len(cols) < 2: return pd.DataFrame()
    X = df[cols].astype(float).fillna(0.0)
    y = X.mean(axis=1)
    model = train_xgb_regressor(X.to_numpy(), y.to_numpy())
    pred = model.predict(X.to_numpy())
    resid = y - pred
    thr = np.percentile(np.abs(resid), 97)
    out = df.copy()
    out["anomaly"] = np.abs(resid) > thr
    return out[out["anomaly"]]

def cluster_basic_torch(X: np.ndarray, k: int = 3, epochs: int = 50) -> np.ndarray:
    if _TORCH_OK:
        n, d = X.shape
        X_t = torch.tensor(X, dtype=torch.float32)
        cent = torch.randn(k, d, requires_grad=True)
        opt = optim.Adam([cent], lr=0.05)
        for _ in range(epochs):
            dist = ((X_t.unsqueeze(1) - cent.unsqueeze(0))**2).sum(2)
            assign = dist.argmin(1)
            loss = sum(((X_t[assign == i] - cent[i])**2).sum() for i in range(k)) / n
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            dist = ((X_t.unsqueeze(1) - cent.unsqueeze(0))**2).sum(2)
            labels = dist.argmin(1).numpy()
        return labels
    # Fallback: k-means NumPy
    rng = np.random.default_rng(42)
    cent = X[rng.choice(len(X), size=k, replace=False)]
    for _ in range(epochs):
        d2 = ((X[:, None, :] - cent[None, :, :])**2).sum(axis=2)
        lab = d2.argmin(axis=1)
        for i in range(k):
            pts = X[lab == i]
            if len(pts):
                cent[i] = pts.mean(axis=0)
    return lab

# ================== DATASET ML (con ATTIVITÀ e SHORT NAMES) ==================
@st.cache_data(ttl=60, show_spinner=False)
def _ml_dataset(schema: str) -> pd.DataFrame:
    rows = []
    tbls = _list_tables(schema)
    for _, r in tbls.iterrows():
        table_name = r["table_name"]
        df = _load_table(schema, table_name)
        if df.empty:
            continue
        dcol = _find_col(df, ALIAS_DATA)
        ocol = _find_col(df, ALIAS_OP)
        acol = _find_col(df, ALIAS_ATT)
        pcol = _find_col(df, ALIAS_POS)
        pc   = _find_col(df, ALIAS_POS_CONF)
        conv = _find_col(df, ALIAS_CONV)
        lavg = _find_col(df, ALIAS_LAV_GEN)
        lavc = _find_col(df, ALIAS_LAV_CONTATTI)
        lavv = _find_col(df, ALIAS_LAV_VARIE)
        inch = _find_col(df, ALIAS_IN_CHIAMATA)
        inatt= _find_col(df, ALIAS_IN_ATTESA)
        cmade= _find_col(df, ALIAS_CALLS_MADE)
        cans = _find_col(df, ALIAS_CALLS_ANSWERED)
        proc = _find_col(df, ALIAS_PROCESSED)
        if not dcol or not ocol:
            continue

        dt = pd.to_datetime(df[dcol], errors="coerce")
        op_short = df[ocol].astype(str).map(_short_name)

        tmp = pd.DataFrame({
            "mese": dt.dt.to_period("M").astype(str),
            "operatore": df[ocol].astype(str),
            "operatore_short": op_short,
            "attivita": df[acol].astype(str) if acol else "UNKNOWN",
            "positivi": _to_numeric_fast(df[pcol]) if pcol else 0,
            "positivi_conf": _to_numeric_fast(df[pc]) if pc else 0,
            "ore_conv": _to_hours(df[conv]) if conv else 0,
            "lavorazione_gen": _to_hours(df[lavg]) if lavg else 0,
            "lav_contatti": _to_hours(df[lavc]) if lavc else 0,
            "lav_varie": _to_hours(df[lavv]) if lavv else 0,
            "in_chiamata": _to_hours(df[inch]) if inch else 0,
            "in_attesa": _to_hours(df[inatt]) if inatt else 0,
            "calls_made": _to_numeric_fast(df[cmade]) if cmade else 0,
            "calls_answered": _to_numeric_fast(df[cans]) if cans else 0,
            "processed": _to_numeric_fast(df[proc]) if proc else 0,
        }).dropna(subset=["mese"])
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()

    all_df = pd.concat(rows, ignore_index=True)

    # Derivate per riga
    all_df["chiamata_tot"] = all_df["ore_conv"].fillna(0) + all_df["in_chiamata"].fillna(0)
    all_df["answer_rate"]  = np.where(all_df["calls_made"]>0, all_df["calls_answered"]/all_df["calls_made"], np.nan)
    all_df["call_load"]    = np.where(all_df["lavorazione_gen"]>0, all_df["chiamata_tot"]/all_df["lavorazione_gen"], np.nan)
    all_df["in_call_share"]= np.where(all_df["lavorazione_gen"]>0, all_df["in_chiamata"]/all_df["lavorazione_gen"], np.nan)
    all_df["conf_rate"]    = np.where(all_df["positivi"]>0, all_df["positivi_conf"]/all_df["positivi"], np.nan)

    return all_df

# ================== GRAFICI BASE ==================
def _scatter_xy_by_op(df: pd.DataFrame, x: str, y: str, title: str, color_map: dict) -> go.Figure:
    fig = go.Figure()
    if not df.empty and {x, y, "operatore_short"}.issubset(df.columns):
        for op in df["operatore_short"].unique():
            sub = df[df["operatore_short"] == op]
            fig.add_trace(go.Scatter(
                x=sub[x], y=sub[y], mode="markers",
                marker=dict(size=9, color=color_map.get(op, BLUE)),
                name=op, showlegend=False
            ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=440, margin=dict(t=60,l=70,r=60,b=120),
        xaxis_title=x, yaxis_title=y, title=title, showlegend=False
    )
    return fig

def _scatter_cat_y_by_op(df: pd.DataFrame, xcat: str, y: str, title: str, color_map: dict) -> go.Figure:
    fig = go.Figure()
    if not df.empty and {xcat, y}.issubset(df.columns):
        order = df.groupby(xcat)[y].sum().sort_values(ascending=False).index.tolist()
        for op in order:
            sub = df[df[xcat] == op]
            fig.add_trace(go.Scatter(
                x=[op]*len(sub), y=sub[y], mode="markers",
                marker=dict(size=9, color=color_map.get(op, BLUE)),
                name=op, showlegend=False
            ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=440, margin=dict(t=60,l=70,r=60,b=160),
        xaxis_title=xcat, yaxis_title=y, title=title, showlegend=False
    )
    fig.update_xaxes(tickangle=-90)
    return fig

# ================== FILTRI ==================
def section_filters(schema: str):
    st.subheader("Filtri")
    df = _ml_dataset(schema)
    mesi = sorted(df["mese"].dropna().unique().tolist()) if not df.empty else []
    att = sorted(df["attivita"].dropna().unique().tolist()) if not df.empty and "attivita" in df.columns else []
    ops = sorted(df["operatore_short"].dropna().unique().tolist()) if not df.empty else []
    c1, c2, c3 = st.columns(3)
    with c1:
        sel_mesi = st.multiselect("Mese (YYYY-MM)", mesi, default=mesi)
    with c2:
        sel_att = st.multiselect("Attività", att, default=att)
    with c3:
        sel_ops = st.multiselect("Operatore", ops, default=ops)
    # Info disponibilità librerie
    lib_info = []
    lib_info.append("XGBoost OK" if _XGB_OK else "XGBoost assente (fallback Ridge)")
    lib_info.append("PyTorch OK" if _TORCH_OK else "PyTorch assente (fallback NumPy)")
    st.caption(" | ".join(lib_info))
    return df, sel_mesi, sel_att, sel_ops

def apply_filters(df: pd.DataFrame, sel_mesi: List[str], sel_att: List[str], sel_ops: List[str]) -> pd.DataFrame:
    if df.empty: return df
    mask = pd.Series(True, index=df.index)
    if sel_mesi:
        mask &= df["mese"].isin(sel_mesi)
    if sel_att and "attivita" in df.columns:
        mask &= df["attivita"].isin(sel_att)
    if sel_ops:
        mask &= df["operatore_short"].isin(sel_ops)
    return df[mask].reset_index(drop=True)

# ================== SEZIONI ANALISI ==================
def section_team_overview(df: pd.DataFrame):
    st.header("Panoramica Team")
    if df.empty:
        st.info("Nessun dato disponibile.")
        return

    agg = (df.groupby("operatore_short", as_index=False)
             .agg(positivi=("positivi","sum"),
                  ore_conv=("ore_conv","sum"),
                  in_chiamata=("in_chiamata","sum"),
                  lavorazione_gen=("lavorazione_gen","sum"),
                  calls_made=("calls_made","sum"),
                  calls_answered=("calls_answered","sum"),
                  processed=("processed","sum")))
    agg["answer_rate"]=np.where(agg["calls_made"]>0,agg["calls_answered"]/agg["calls_made"],np.nan)
    agg["in_call_share"]=np.where(agg["lavorazione_gen"]>0,agg["in_chiamata"]/agg["lavorazione_gen"],np.nan)
    agg["call_load"]=np.where(agg["lavorazione_gen"]>0,(agg["ore_conv"]+agg["in_chiamata"])/agg["lavorazione_gen"],np.nan)

    cmap = _op_color_map(agg["operatore_short"].tolist())

    fig = _scatter_cat_y_by_op(agg, "operatore_short", "positivi", "Positivi per operatore (totale periodo)", cmap)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.markdown(_legend_ops_html(agg["operatore_short"].tolist(), cmap), unsafe_allow_html=True)
    st.caption("Ogni pallino è un operatore; più in alto = più positivi nel periodo filtrato.")

    st.dataframe(agg, use_container_width=True)

def section_explainability(df: pd.DataFrame):
    st.header("Explainability e priorità")
    st.caption("Feature collegate ai positivi ed effetto medio atteso variando ogni metrica.")

    feats = [c for c in [
        "lavorazione_gen","ore_conv","in_chiamata","in_attesa","chiamata_tot",
        "calls_made","calls_answered","processed",
        "answer_rate","in_call_share","call_load"
    ] if c in df.columns]

    if not feats or "positivi" not in df.columns:
        st.info("Dati insufficienti per stime di importanza.")
        return

    X = df[feats].astype(float).fillna(0.0).to_numpy()
    y = df["positivi"].astype(float).fillna(0.0).to_numpy()

    model = train_xgb_regressor(X, y)
    importances = getattr(model, "feature_importances_", np.zeros(len(feats)))
    imp_df = pd.DataFrame({"feature": feats, "importance": importances}).sort_values("importance", ascending=False)

    fig_imp = go.Figure(go.Bar(x=imp_df["feature"], y=imp_df["importance"], marker_color=VIOLET, showlegend=False))
    fig_imp.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                          height=420, margin=dict(t=40, l=70, r=60, b=120),
                          title="Importanza feature (XGBoost o Ridge fallback)",
                          xaxis_title="Feature", yaxis_title="Importanza")
    fig_imp.update_xaxes(tickangle=-90)
    st.plotly_chart(fig_imp, use_container_width=True, config={"displaylogo": False})

    # PDP semplice su 3 variabili
    cand = [c for c in ["ore_conv","in_chiamata","calls_made","processed","answer_rate","call_load"] if c in df.columns][:3]
    if cand:
        for c in cand:
            xvals = df[c].dropna().astype(float)
            if xvals.empty:
                continue
            grid = np.linspace(np.nanpercentile(xvals, 5), np.nanpercentile(xvals, 95), 25)
            base = df[feats].astype(float).fillna(0.0).copy()
            y_avg = []
            for v in grid:
                base[c] = v
                y_avg.append(float(np.mean(model.predict(base.to_numpy()))))
            figp = go.Figure(go.Scatter(x=grid, y=y_avg, mode="lines+markers", showlegend=False))
            figp.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                               height=420, margin=dict(t=40,l=70,r=60,b=90),
                               title=f"PDP – effetto medio di {c} su positivi",
                               xaxis_title=c, yaxis_title="Predizione media")
            if c in {"answer_rate","in_call_share"}:
                figp.update_xaxes(range=[0,1], tickformat=".0%")
            st.plotly_chart(figp, use_container_width=True, config={"displaylogo": False})

def section_anomalies_and_clusters(df: pd.DataFrame):
    st.header("Anomalie e clustering")
    st.caption("Outlier operativi e gruppi di operatori simili.")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    out = detect_anomalies_xgb(df, num_cols)
    if out.empty:
        st.info("Nessuna anomalia forte rilevata (soglia 97° percentile residui).")
    else:
        cols_show = [c for c in ["mese","operatore_short","attivita","ore_conv","in_chiamata","lavorazione_gen","calls_made","processed","positivi"] if c in out.columns]
        st.dataframe(out[cols_show], use_container_width=True)

    feats_cand = [c for c in ["ore_conv","in_chiamata","in_attesa","lavorazione_gen","positivi","calls_made","processed","answer_rate","call_load","in_call_share"] if c in df.columns]
    agg = df.groupby("operatore_short", as_index=False).agg({c: "mean" for c in feats_cand})
    if len(agg) >= 3 and agg.shape[1] >= 3:
        ops = agg["operatore_short"].tolist()
        cmap = _op_color_map(ops)
        Xc = agg.drop(columns=["operatore_short"]).astype(float).fillna(0.0).to_numpy()
        labs = cluster_basic_torch(Xc, k=min(3, len(agg)))
        agg["cluster"] = labs

        if {"ore_conv","in_chiamata"}.issubset(agg.columns):
            figc = go.Figure()
            for _, r in agg.iterrows():
                figc.add_trace(go.Scatter(
                    x=[r["ore_conv"]], y=[r["in_chiamata"]],
                    mode="markers",
                    marker=dict(size=10, color=cmap[r["operatore_short"]]),
                    showlegend=False,
                    name=r["operatore_short"]
                ))
            figc.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                               height=460, margin=dict(t=60,l=70,r=60,b=120),
                               xaxis_title="ore_conv", yaxis_title="in_chiamata",
                               title="Clustering operatori (profilo voce)", showlegend=False)
            st.plotly_chart(figc, use_container_width=True, config={"displaylogo": False})
            st.markdown(_legend_ops_html(ops, cmap), unsafe_allow_html=True)
        st.dataframe(agg[["operatore_short","cluster"] + feats_cand], use_container_width=True)
    else:
        st.info("Operatori o feature insufficienti per clustering.")

def section_efficiency(df: pd.DataFrame):
    st.header("Efficienza operativa")
    st.caption("Positivi per ora voce = output normalizzato al tempo speso in conversazione.")
    if {"ore_conv","in_chiamata","positivi","operatore_short"}.issubset(df.columns):
        base = df.copy()
        base["ore_voce"] = base["ore_conv"].fillna(0) + base["in_chiamata"].fillna(0)
        agg = base.groupby("operatore_short", as_index=False).agg(pos=("positivi","sum"), ore_voce=("ore_voce","sum"))
        agg["pos_per_hour"] = np.where(agg["ore_voce"]>0, agg["pos"]/agg["ore_voce"], np.nan)

        ops = agg["operatore_short"].tolist()
        cmap = _op_color_map(ops)

        fig = go.Figure()
        for _, r in agg.iterrows():
            fig.add_trace(go.Scatter(
                x=[r["ore_voce"]], y=[r["pos_per_hour"]],
                mode="markers", marker=dict(size=10, color=cmap[r["operatore_short"]]),
                showlegend=False, name=r["operatore_short"]
            ))
        fig.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                          height=460, margin=dict(t=70,l=70,r=60,b=120),
                          xaxis_title="Ore voce (conv + in_chiamata)", yaxis_title="Positivi per ora voce",
                          showlegend=False, title="Frontiera di efficienza")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.markdown(_legend_ops_html(ops, cmap), unsafe_allow_html=True)
    else:
        st.info("Servono: ore_conv, in_chiamata, positivi.")

# ================== EXPORT / HOME ==================
def _simple_html(schema: str, sel_mesi: List[str], sel_att: List[str], sel_ops: List[str]) -> bytes:
    html = f"""
<html><head><meta charset='utf-8'><title>SDR Insight ML</title></head>
<body style='font-family:Arial; color:#111'>
<h2>SDR Insight ML</h2>
<p>Schema attivo: <b>{schema}</b></p>
<p>Filtri: Mesi={sel_mesi} | Attività={sel_att} | Operatori={sel_ops}</p>
<p>Questa esportazione contiene un riepilogo testuale. I grafici restano nell'app Streamlit.</p>
</body></html>
"""
    return html.encode("utf-8")

def go_home():
    st.session_state["page"] = "home"
    try:
        st.rerun()
    except Exception:
        pass

# ================== MAIN ==================
def render():
    schema = _current_db()

    st.title("SDR Insight ML")
    st.caption(f"Schema attivo: {schema}")

    df, sel_mesi, sel_att, sel_ops = section_filters(schema)
    df = apply_filters(df, sel_mesi, sel_att, sel_ops)

    section_team_overview(df)
    section_explainability(df)
    section_anomalies_and_clusters(df)
    section_efficiency(df)

    st.markdown("---")
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
            data=_simple_html(schema, sel_mesi, sel_att, sel_ops),
            file_name=f"SDR_Insight_ML_{int(time.time())}.html",
            mime="text/html",
            use_container_width=True,
        )
    with c2:
        if st.button("🏠 Home", use_container_width=True):
            go_home()

if __name__ == "__main__":
    render()
