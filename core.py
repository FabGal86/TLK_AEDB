# ======================================================
# core.py  (compatibile Python 3.13) – versione snella
# ======================================================

from __future__ import annotations
import os, io, math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# --- Plot opzionali -------------------------------------------------
try:
    import plotly.express as px
    import plotly.graph_objects as go
    _has_plotly = True
except Exception:
    px = go = None
    _has_plotly = False

try:
    import matplotlib.pyplot as plt  # type: ignore
    _has_mpl = True
except Exception:
    plt = None  # type: ignore
    _has_mpl = False

# --- Config Streamlit -----------------------------------------------
st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# --- SQL config -----------------------------------------------------
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

@dataclass
class Config:
    sql_url: str | None = None

def load_config() -> Config:
    # precedence: st.secrets -> env
    url = None
    try:
        url = st.secrets.get("SQLALCHEMY_URL")  # type: ignore[attr-defined]
    except Exception:
        url = None
    if not url:
        url = os.getenv("SQLALCHEMY_URL")
    return Config(sql_url=url)

CFG = load_config()

@st.cache_resource(show_spinner=False)
def get_engine() -> Engine | None:
    if not CFG.sql_url:
        return None
    # Esempio URL: mysql+pymysql://streamlit:pwd@host:3306/AEDBdata?charset=utf8mb4
    return create_engine(
        CFG.sql_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        future=True,
    )

def sql_read(query: str, params: dict | None = None) -> pd.DataFrame:
    eng = get_engine()
    if eng is None:
        raise RuntimeError("Nessun URL SQL configurato. Imposta SQLALCHEMY_URL nei Secrets.")
    with eng.connect() as cx:
        return pd.read_sql(text(query), cx, params=params or {})

def sql_write(df: pd.DataFrame, table: str, if_exists: str = "append", index: bool = False) -> None:
    eng = get_engine()
    if eng is None:
        raise RuntimeError("Nessun URL SQL configurato. Imposta SQLALCHEMY_URL nei Secrets.")
    df.to_sql(table, eng, if_exists=if_exists, index=index)

# --- Utilità leggere -----------------------------------------------
def kpi(label: str, value: Any, fmt: str | None = None) -> None:
    if fmt and isinstance(value, (int, float, np.number)):
        st.metric(label, fmt.format(value))
    else:
        st.metric(label, value)

def df_info(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null": df.notna().sum().values,
        "null_%": (df.isna().mean() * 100).round(2).values,
        "unique": df.nunique().values,
    })

def iqr_outliers(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return (s < low) | (s > high)

def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)

# --- ML base senza sklearn -----------------------------------------
def linreg_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    X1 = np.column_stack([X, np.ones(len(X))])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return beta[:-1], float(beta[-1])

def linreg_predict(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    X = np.asarray(X, float)
    return X @ coef + intercept

def standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean(axis=0)) / (x.std(axis=0, ddof=0) + 1e-12)

def minmax_scale(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.min(axis=0)) / (x.ptp(axis=0) + 1e-12)

# --- Forecast lineare semplice -------------------------------------
def forecast_series(df: pd.DataFrame, date_col: str, y_col: str, horizon: int = 30, freq: str = "D") -> pd.DataFrame:
    df = df[[date_col, y_col]].dropna().sort_values(date_col).copy()
    df["t"] = np.arange(len(df))
    coef, intercept = linreg_fit(df[["t"]].values, df[y_col].values)
    fut_t = np.arange(len(df) + horizon)
    yhat = linreg_predict(fut_t.reshape(-1, 1), coef, intercept)
    dates = pd.date_range(df[date_col].min(), periods=len(df) + horizon, freq=freq)
    resid = df[y_col].values - linreg_predict(df[["t"]].values, coef, intercept)
    sigma = float(np.std(resid, ddof=0))
    return pd.DataFrame({
        "date": dates,
        "yhat": yhat,
        "yhat_lower": yhat - 1.96 * sigma,
        "yhat_upper": yhat + 1.96 * sigma,
    })

# --- Plot helpers con fallback -------------------------------------
def plot_ts(df: pd.DataFrame, x: str, y: str, title: str | None = None):
    if _has_plotly:
        fig = px.line(df, x=x, y=y, title=title)
        st.plotly_chart(fig, use_container_width=True)
    elif _has_mpl:
        fig, ax = plt.subplots()
        df.plot(x=x, y=y, ax=ax, title=title or "")
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Nessuna libreria grafica disponibile.")

def plot_bar(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str | None = None):
    if _has_plotly:
        fig = px.bar(df, x=x, y=y, color=color, title=title, text_auto=".2f", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    elif _has_mpl:
        fig, ax = plt.subplots()
        if color and color in df.columns:
            for k, g in df.groupby(color):
                ax.bar(g[x], g[y], label=str(k))
            ax.legend()
        else:
            ax.bar(df[x], df[y])
        ax.set_title(title or "")
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Nessuna libreria grafica disponibile.")

# --- Download CSV ---------------------------------------------------
def to_csv_download(df: pd.DataFrame, filename: str = "data.csv"):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("📥 Scarica CSV", buf.getvalue().encode("utf-8"),
                       file_name=filename, mime="text/csv")
