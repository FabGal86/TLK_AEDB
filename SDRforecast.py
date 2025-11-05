# SDRforecast.py – Python 3.13, 8 modelli (4 classici, 4 ML/DL)
# Grafico giornaliero, mesi colorati, cut-off 120gg, forecast settimanale (fino a 50 settimane)
from __future__ import annotations

import re, warnings, math, time, io, datetime as dt
from typing import List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from pathlib import Path

# dipendenze opzionali
try:
    import statsmodels.api as sm
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    sm = None
    ExponentialSmoothing = None

try:
    import xgboost as xgb
except Exception:
    xgb = None

# PyTorch opzionale, con guardia robusta
try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore
    nn = None     # type: ignore
    _TORCH_OK = False

# ================== CONFIG ==================
FALLBACK_SCHEMA = "aedbdata"
ROW_LIMIT_PER_TABLE = 0
DARK_BG = "#0f1113"

ALIAS_DATA = ["data","date","timestamp","datetime"]
ALIAS_POS  = ["positivi","positive","lead_positivi","esito_positivo"]
ALIAS_ATT  = ["attivita","attività","activity"]
ALIAS_OP   = ["operatore","operator","utente","user"]

# palette 12 mesi
MONTH_COLORS = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728",
    "#9467bd","#8c564b","#e377c2","#7f7f7f",
    "#bcbd22","#17becf","#a55194","#6b6ecf"
]

# ================== DB-FREE / CSV / EMBEDDED HELPERS ==================
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
    return re.sub(r"\W+", "", (s or "").lower())

def _find_col(df: pd.DataFrame, aliases: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    norm_map = {_norm(c): c for c in df.columns}
    for a in aliases:
        na = _norm(a)
        if na in norm_map:
            return norm_map[na]
    for a in aliases:
        na = _norm(a)
        for nc, real in norm_map.items():
            if na in nc:
                return real
    return None

def _to_datetime_fast(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce")

def _to_numeric_fast(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _rmse_adjacent(y: np.ndarray) -> Optional[float]:
    if len(y) <= 1:
        return None
    return float(np.sqrt(np.mean((y[1:] - y[:-1])**2)))

def _casefold_eq(series: pd.Series, value: str) -> pd.Series:
    return series.astype(str).str.strip().str.casefold() == value.strip().casefold()

def go_home():
    st.session_state["page"] = "home"

# ================== LISTE VALORI FILTRI ==================
@st.cache_data(ttl=60, show_spinner=False)
def _list_unique_values(schema: str, aliases: List[str]) -> List[str]:
    tbls = _list_tables(schema)
    values = set()
    for _, r in tbls.iterrows():
        df = _load_table(schema, r["table_name"])
        if df is None or df.empty:
            continue
        col = _find_col(df, aliases)
        if not col:
            continue
        vals = pd.Series(df[col].dropna().unique()).astype(str)
        for v in vals:
            v = v.strip()
            if v:
                values.add(v)
    return sorted(values, key=lambda x: x.casefold())

# ================== COSTRUZIONE SERIE ==================
@st.cache_data(ttl=60, show_spinner=False)
def _build_daily_series(schema: str,
                        filtro_attivita: Optional[str] = None,
                        filtro_operatore: Optional[str] = None) -> pd.DataFrame:
    """Aggrega a giorno le colonne 'positivi' sommandole, filtrando dove possibile."""
    tbls = _list_tables(schema)
    rows = []
    for _, r in tbls.iterrows():
        df = _load_table(schema, r["table_name"])
        if df is None or df.empty:
            continue

        dcol = _find_col(df, ALIAS_DATA)
        pcol = _find_col(df, ALIAS_POS)
        acol = _find_col(df, ALIAS_ATT)
        ocol = _find_col(df, ALIAS_OP)

        if filtro_attivita and filtro_attivita != "Tutte":
            if not acol:
                continue
            df = df[_casefold_eq(df[acol], filtro_attivita)]
        if filtro_operatore and filtro_operatore != "Tutti":
            if not ocol:
                continue
            df = df[_casefold_eq(df[ocol], filtro_operatore)]

        if df.empty or not dcol or not pcol:
            continue

        dtv = _to_datetime_fast(df[dcol]).dt.normalize()
        pos = _to_numeric_fast(df[pcol])
        tmp = pd.DataFrame({"giorno": dtv, "positivi": pos}).dropna()
        if not tmp.empty:
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["giorno","positivi"])

    all_df = pd.concat(rows, ignore_index=True)
    agg = (all_df.groupby("giorno", as_index=False)["positivi"].sum()
           .sort_values("giorno").reset_index(drop=True))
    agg = agg[agg["positivi"].astype(float).notna()]
    return agg

# ================== MODELLI ==================
ModelName = Literal[
    "naive",
    "drift",
    "linreg",
    "holtwinters",
    "xgb_lag",
    "xgb_lag_cal",
    "torch_mlp",
    "torch_lstm"
]

def _pack_future_days(last_dt: pd.Timestamp, horizon_days: int, y: np.ndarray) -> pd.DataFrame:
    idx = [(last_dt + pd.Timedelta(days=h)) for h in range(1, horizon_days+1)]
    return pd.DataFrame({"giorno": idx, "forecast": y})

def _linreg_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    x = np.arange(len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    x_future = np.arange(len(y), len(y) + horizon)
    return a * x_future + b

def _holtwinters_forecast(y: np.ndarray, horizon: int) -> np.ndarray:
    if ExponentialSmoothing is None:
        raise RuntimeError("statsmodels non disponibile per Holt-Winters.")
    seasonal_periods = 7 if len(y) >= 14 else None
    if seasonal_periods:
        model = ExponentialSmoothing(
            y, trend="add", seasonal="add", seasonal_periods=seasonal_periods, initialization_method="estimated"
        ).fit(optimized=True)
    else:
        model = ExponentialSmoothing(
            y, trend="add", seasonal=None, initialization_method="estimated"
        ).fit(optimized=True)
    fc = model.forecast(horizon)
    return np.asarray(fc, dtype=float)

# ---------- Feature engineering per ML/DL ----------
def _make_supervised(y: np.ndarray, n_lags: int, add_cal: bool,
                     start_dt: pd.Timestamp, cal_period: int = 7) -> tuple[np.ndarray,np.ndarray]:
    y = np.asarray(y, dtype=float)
    X_list, tgt = [], []
    for t in range(n_lags, len(y)):
        row = [y[t - i - 1] for i in range(n_lags)]
        if add_cal:
            ts = (start_dt + pd.Timedelta(days=t))
            k = (ts.dayofweek % cal_period) if cal_period == 7 else (ts.dayofyear % cal_period)
            row += [math.sin(2*math.pi*k/cal_period), math.cos(2*math.pi*k/cal_period)]
        X_list.append(row)
        tgt.append(y[t])
    X = np.asarray(X_list, dtype=float)
    y_out = np.asarray(tgt, dtype=float)
    return X, y_out

def _recursive_predict(initial: list[float], horizon: int, predict_fn, with_cal: bool,
                       start_dt: pd.Timestamp, cal_period: int = 7) -> np.ndarray:
    hist = list(initial)
    n_lags = len(initial)
    out = []
    for _ in range(horizon+0):
        row = hist[-n_lags:][:]
        if with_cal:
            ts = (start_dt + pd.Timedelta(days=len(hist)))
            k = (ts.dayofweek % cal_period) if cal_period == 7 else (ts.dayofyear % cal_period)
            row += [math.sin(2*math.pi*k/cal_period), math.cos(2*math.pi*k/cal_period)]
        X = np.asarray(row, dtype=float).reshape(1, -1)
        yhat = float(predict_fn(X))
        out.append(yhat)
        hist.append(yhat)
        if len(out) >= horizon:
            break
    return np.asarray(out, dtype=float)

# ---------- XGBoost ----------
def _xgb_forecast(y: np.ndarray, horizon: int, n_lags: int, with_calendar: bool,
                  start_dt: pd.Timestamp) -> np.ndarray:
    if xgb is None:
        raise RuntimeError("xgboost non disponibile.")
    X, y_sup = _make_supervised(y, n_lags=n_lags, add_cal=with_calendar, start_dt=start_dt, cal_period=7)
    if len(y_sup) < 5:
        return np.repeat(y[-1], horizon)
    model = xgb.XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=0
    )
    model.fit(X, y_sup)
    seed = list(y[-n_lags:])
    start_for_rec = start_dt
    return _recursive_predict(seed, horizon, model.predict, with_calendar, start_for_rec, cal_period=7)

# ---------- PyTorch ----------
def _torch_set_seed(seed: int = 42) -> None:
    if not _TORCH_OK:
        return
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass

# Definisci le reti solo se Torch è disponibile
if _TORCH_OK:
    class _TorchMLP(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        def forward(self, x):
            return self.net(x)

    class _TorchLSTM(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
            self.head = nn.Linear(32, 1)
            self.in_dim = in_dim
        def forward(self, x_seq):  # (B, T, 1)
            out, _ = self.lstm(x_seq)
            return self.head(out[:, -1, :])
else:
    class _TorchMLP:  # type: ignore
        pass
    class _TorchLSTM:  # type: ignore
        pass

def _torch_mlp_forecast(y: np.ndarray, horizon: int, n_lags: int, with_calendar: bool,
                        start_dt: pd.Timestamp) -> np.ndarray:
    if not _TORCH_OK:
        raise RuntimeError("PyTorch non disponibile.")
    X, y_sup = _make_supervised(y, n_lags=n_lags, add_cal=with_calendar, start_dt=start_dt, cal_period=7)
    if len(y_sup) < 5:
        return np.repeat(y[-1], horizon)
    _torch_set_seed(42)
    device = torch.device("cpu")
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_sup.reshape(-1,1), dtype=torch.float32, device=device)

    model = _TorchMLP(in_dim=X.shape[1]).to(device)  # type: ignore[arg-type]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 600 if len(y_sup) > 100 else 300
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

    model.eval()
    def _predict(Xnp: np.ndarray) -> float:
        with torch.no_grad():
            xt = torch.tensor(Xnp, dtype=torch.float32)
            yhat = model(xt).cpu().numpy().ravel()[0]
        return yhat

    seed = list(y[-n_lags:])
    start_for_rec = start_dt
    return _recursive_predict(seed, horizon, _predict, with_calendar, start_for_rec, cal_period=7)

def _torch_lstm_forecast(y: np.ndarray, horizon: int, n_lags: int, start_dt: pd.Timestamp) -> np.ndarray:
    if not _TORCH_OK:
        raise RuntimeError("PyTorch non disponibile.")
    y = np.asarray(y, dtype=float)
    X_list, tgt = [], []
    for t in range(n_lags, len(y)):
        X_list.append(y[t-n_lags:t].reshape(-1,1))
        tgt.append(y[t])
    if len(tgt) < 5:
        return np.repeat(y[-1], horizon)

    _torch_set_seed(42)
    device = torch.device("cpu")
    X_np = np.stack(X_list, axis=0)  # (N, T, 1)
    y_np = np.asarray(tgt, dtype=float).reshape(-1,1)

    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=device)

    model = _TorchLSTM(in_dim=n_lags).to(device)  # type: ignore[arg-type]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 700 if len(tgt) > 120 else 350
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        opt.step()

    model.eval()
    seq = list(y[-n_lags:])
    out = []
    with torch.no_grad():
        for _ in range(horizon):
            x = torch.tensor(np.array(seq).reshape(1, n_lags, 1), dtype=torch.float32, device=device)
            yhat = float(model(x).cpu().numpy().ravel()[0])
            out.append(yhat)
            seq = seq[1:] + [yhat]
    return np.asarray(out, dtype=float)

# ---------- Dispatch ----------
def forecast_dispatch_days(team_daily: pd.DataFrame, horizon_days: int = 7*4,
                           model: ModelName = "linreg",
                           n_lags: int = 14) -> Tuple[pd.DataFrame, Optional[float], str]:
    df = team_daily.copy()
    df = df.dropna(subset=["giorno"]).sort_values("giorno")
    y = df["positivi"].astype(float).to_numpy()
    if len(y) == 0:
        return pd.DataFrame(columns=["giorno","forecast"]), None, "Serie vuota."
    last_dt = pd.to_datetime(df["giorno"].iloc[-1])
    start_dt = pd.to_datetime(df["giorno"].iloc[0])

    note = ""
    if model == "naive":
        fc = np.repeat(y[-1], horizon_days)
        note = "Naive: ripete l'ultimo valore."
    elif model == "drift":
        slope = (y[-1] - y[0]) / max(1, len(y)-1)
        fc = np.array([y[-1] + slope*h for h in range(1, horizon_days+1)])
        note = "Drift: trend lineare dai capi serie."
    elif model == "linreg":
        fc = _linreg_forecast(y, horizon_days)
        note = "Regressione lineare sui tempi."
    elif model == "holtwinters":
        try:
            fc = _holtwinters_forecast(y, horizon_days)
            note = "Holt-Winters additivo; stagionalità 7 se disponibile."
        except Exception as e:
            warnings.warn(str(e))
            fc = _linreg_forecast(y, horizon_days)
            note = "Fallback a linreg: Holt-Winters non disponibile."
    elif model == "xgb_lag":
        try:
            fc = _xgb_forecast(y, horizon_days, n_lags=n_lags, with_calendar=False, start_dt=start_dt)
            note = f"XGBoost regressore con {n_lags} lag."
        except Exception as e:
            warnings.warn(str(e))
            fc = _linreg_forecast(y, horizon_days)
            note = "Fallback a linreg: XGBoost non disponibile."
    elif model == "xgb_lag_cal":
        try:
            fc = _xgb_forecast(y, horizon_days, n_lags=n_lags, with_calendar=True, start_dt=start_dt)
            note = f"XGBoost con {n_lags} lag + feature calendario settimanali."
        except Exception as e:
            warnings.warn(str(e))
            fc = _linreg_forecast(y, horizon_days)
            note = "Fallback a linreg: XGBoost non disponibile."
    elif model == "torch_mlp":
        try:
            fc = _torch_mlp_forecast(y, horizon_days, n_lags=n_lags, with_calendar=True, start_dt=start_dt)
            note = f"PyTorch MLP con {n_lags} lag + feature calendario settimanali."
        except Exception as e:
            warnings.warn(str(e))
            fc = _linreg_forecast(y, horizon_days)
            note = "Fallback a linreg: PyTorch non disponibile."
    elif model == "torch_lstm":
        try:
            fc = _torch_lstm_forecast(y, horizon_days, n_lags=n_lags, start_dt=start_dt)
            note = f"PyTorch LSTM con finestra {n_lags}."
        except Exception as e:
            warnings.warn(str(e))
            fc = _linreg_forecast(y, horizon_days)
            note = "Fallback a linreg: PyTorch non disponibile."
    else:
        fc = _linreg_forecast(y, horizon_days)
        note = "Modello non riconosciuto. Uso linreg."

    rmse = _rmse_adjacent(y)
    return _pack_future_days(last_dt, horizon_days, np.asarray(fc, dtype=float)), rmse, note

# ================== PAGINA STREAMLIT ==================
def render() -> None:
    st.title("SDR Forecast – grafico giornaliero con mesi colorati")
    schema = _current_db()
    st.caption(f"Schema attivo: {schema}")

    # ---- Filtri UI ----
    all_attivita = _list_unique_values(schema, ALIAS_ATT)
    all_operatore = _list_unique_values(schema, ALIAS_OP)
    attivita_sel = st.selectbox("Filtro Attività", ["Tutte"] + all_attivita, index=0)
    operatore_sel = st.selectbox("Filtro Operatore", ["Tutti"] + all_operatore, index=0)

    team_full = _build_daily_series(
        schema,
        filtro_attivita=None if attivita_sel == "Tutte" else attivita_sel,
        filtro_operatore=None if operatore_sel == "Tutti" else operatore_sel,
    )
    if team_full.empty:
        st.warning("Nessuna serie giornaliera disponibile con i filtri correnti.")
        return

    # ---- Cut-off e modello ----
    st.subheader("Impostazioni modello")
    cutoff_offset = st.number_input("Cut-off offset (giorni, min -120, max 0)", min_value=-120, max_value=0, value=0, step=1)

    models = [
        ("linreg", "Regressione lineare (trad.)"),
        ("naive", "Naive (trad.)"),
        ("drift", "Drift (trad.)"),
        ("holtwinters", "Holt-Winters (trad.)"),
        ("xgb_lag", "XGBoost lag"),
        ("xgb_lag_cal", "XGBoost lag+calendario"),
        ("torch_mlp", "PyTorch MLP"),
        ("torch_lstm", "PyTorch LSTM"),
    ]
    model_keys = [m[0] for m in models]
    model_labels = {k:v for k,v in models}

    model_name = st.selectbox("Modello", model_keys, index=0, format_func=lambda k: model_labels[k])
    weeks = st.number_input("Orizzonte (settimane)", 1, 50, 4)
    n_lags = st.number_input("Lag per modelli ML/DL (giorni)", 3, 60, 14)

    # costruiamo train fino al cut-off
    tmp = team_full.copy()
    tmp["_dt"] = pd.to_datetime(tmp["giorno"]).dt.normalize()
    last_dt = tmp["_dt"].max()
    cutoff_dt = (last_dt + pd.Timedelta(days=int(cutoff_offset)))
    team_train = tmp[tmp["_dt"] <= cutoff_dt][["giorno","positivi"]].reset_index(drop=True)
    if team_train.empty:
        team_train = tmp[["giorno","positivi"]].iloc[[0]].reset_index(drop=True)

    horizon_days = int(weeks) * 7
    fut_daily, rmse, note = forecast_dispatch_days(team_train.rename(columns={"giorno":"giorno"}),
                                                   horizon_days=horizon_days,
                                                   model=model_name,
                                                   n_lags=int(n_lags))

    # ---- Grafico giornaliero con mesi colorati + trendline ----
    fig = go.Figure()
    color_forecast = "#d62728"
    grid_margin = dict(l=20, r=20, t=40, b=40)

    # storico per mese: segmenti colorati
    hist = tmp[["_dt","positivi"]].sort_values("_dt").copy()
    hist["anno_mese"] = hist["_dt"].dt.to_period("M").astype(str)
    months = hist["anno_mese"].unique().tolist()
    for i, m in enumerate(months):
        seg = hist[hist["anno_mese"] == m]
        if i > 0:
            prev = hist[hist["anno_mese"] == months[i-1]].iloc[-1:][["_dt","positivi"]]
            seg = pd.concat([prev, seg], ignore_index=True)
        fig.add_trace(go.Scatter(
            x=seg["_dt"], y=seg["positivi"],
            mode="lines",
            name=m if i == 0 else None,
            line=dict(color=MONTH_COLORS[i % len(MONTH_COLORS)]),
            showlegend=(i == 0)
        ))

    # forecast giornaliero
    fc = None
    if not fut_daily.empty:
        start_fc = (pd.to_datetime(cutoff_dt) + pd.Timedelta(days=1)).normalize()
        fc = fut_daily.copy()
        fc["_dt"] = pd.to_datetime(fc["giorno"]).dt.normalize()
        fc = fc[fc["_dt"] >= start_fc]
        if not fc.empty:
            fig.add_trace(go.Scatter(
                x=fc["_dt"], y=fc["forecast"],
                mode="lines",
                name="Forecast",
                line=dict(color=color_forecast)
            ))

    # trendline lineare completa
    if not hist.empty and hist["positivi"].notna().sum() >= 2:
        x_hist = hist["_dt"].map(pd.Timestamp.toordinal).to_numpy()
        y_hist = hist["positivi"].astype(float).to_numpy()
        A = np.vstack([x_hist, np.ones(len(x_hist))]).T
        a, b = np.linalg.lstsq(A, y_hist, rcond=None)[0]
        x0_dt = hist["_dt"].min()
        x1_dt = (fc["_dt"].max() if fc is not None and not fc.empty else hist["_dt"].max())
        y0 = a * x0_dt.toordinal() + b
        y1 = a * x1_dt.toordinal() + b
        fig.add_trace(go.Scatter(
            x=[x0_dt, x1_dt], y=[y0, y1],
            mode="lines",
            name="Trendline",
            line=dict(color="#cccccc", dash="dot")
        ))

    # linea verticale di cut-off
    x_cut = pd.to_datetime(cutoff_dt).to_pydatetime()
    fig.add_shape(type="line", x0=x_cut, x1=x_cut, y0=0, y1=1, xref="x", yref="paper",
                  line=dict(color="#888", dash="dash"))
    fig.add_annotation(x=x_cut, y=1, yref="paper", text="cut-off", showarrow=False, xanchor="left", yshift=10)

    fig.update_xaxes(title_text="Data")
    fig.update_yaxes(title_text="Positivi")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        height=520, margin=grid_margin,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # metriche e note
    st.metric("RMSE adiacente (proxy volatilità)", f"{rmse:.2f}" if rmse is not None else "n/d")
    st.caption(note)

    with st.expander("Dati aggregati giornalieri (post-filtri)"):
        st.dataframe(team_full[["giorno","positivi"]], use_container_width=True)

    with st.expander("Forecast giornaliero dettagliato"):
        st.dataframe(fut_daily, use_container_width=True)

    # ================== BOTTONI FUXIA ==================
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

    # Export HTML
    try:
        fig_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
    except Exception:
        fig_html = "<p>Impossibile serializzare il grafico.</p>"

    rmse_txt = f"{rmse:.2f}" if rmse is not None else "n/d"
    today_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    html_page = f"""<!DOCTYPE html>
<html lang="it"><head><meta charset="utf-8">
<title>SDR Forecast Export</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{ background:#0f1113; color:#eaeaea; font-family:Arial,Helvetica,sans-serif; padding:20px; }}
h2,h3 {{ margin:10px 0; }}
pre {{ white-space:pre-wrap; }}
hr {{ border:1px solid #333; }}
</style></head>
<body>
<h2>SDR Forecast – Export</h2>
<p><b>Schema:</b> {schema} &nbsp;|&nbsp; <b>Modello:</b> {model_name} &nbsp;|&nbsp; <b>RMSE adj:</b> {rmse_txt} &nbsp;|&nbsp; <b>Gen:</b> {today_str}</p>
<div>{fig_html}</div>
<hr/>
<h3>Dati (prime righe)</h3>
<pre>{team_full.head(30).to_string(index=False)}</pre>
<h3>Forecast (prime righe)</h3>
<pre>{fut_daily.head(42).to_string(index=False) if not fut_daily.empty else "n/d"}</pre>
</body></html>"""

    buf = io.BytesIO(html_page.encode("utf-8"))

    c1b, c2b = st.columns(2)
    with c1b:
        st.download_button(
            "⬇️ Scarica HTML",
            data=buf,
            file_name=f"SDR_Forecast_{schema}_{int(time.time())}.html",
            mime="text/html",
            key="sdrf_html",
            use_container_width=True,
        )
    with c2b:
        if st.button("🏠 Home", key="sdrf_home", use_container_width=True):
            go_home()

if __name__ == "__main__":
    render()
