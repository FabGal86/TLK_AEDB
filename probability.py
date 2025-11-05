#!/usr/bin/env python3
# probability.py
# Monte Carlo per stimare la probabilità di superare soglie di crescita cumulata
# e la probabilità di subire cali delle stesse percentuali nello stesso arco temporale.
# Aggiunge confronto MC vs analitico (quando scipy è disponibile) e indicazione
# del posizionamento attuale (aspettativa di crescita cumulata).
# Nuovo: grafici che comparano probabilità di crescita e decrescita rispetto al valore attuale.

from __future__ import annotations

import argparse
from typing import Any, Optional, Dict, List, Tuple

import math
import numpy as np
import pandas as pd

THRESHOLDS = [1.025, 1.05, 1.10]
HORIZONS = [30, 60, 90, 120]


# ---------- Monte Carlo core ----------

def simulate_probs(mu: float, sigma: float, trials: int,
                   horizons: List[int] = HORIZONS, thresholds: List[float] = THRESHOLDS,
                   seed: Optional[int] = 42) -> Dict[int, Dict[float, float]]:
    """
    Monte Carlo per probabilità di CRESCITA: P(crescita cumulata >= threshold).
    Ritorna dizionario: {h: {threshold: prob}}.
    """
    rng = np.random.default_rng(seed)
    T_max = int(max(horizons))
    shocks = rng.normal(loc=mu, scale=sigma, size=(T_max, int(trials)))
    cum_log = np.cumsum(shocks, axis=0)
    results: Dict[int, Dict[float, float]] = {}
    for h in horizons:
        idx = int(h) - 1
        cum_growth = np.exp(cum_log[idx, :])
        results[int(h)] = {float(thr): float(np.mean(cum_growth >= thr)) for thr in thresholds}
    return results


def simulate_probs_decline(mu: float, sigma: float, trials: int,
                           horizons: List[int] = HORIZONS, thresholds: List[float] = THRESHOLDS,
                           seed: Optional[int] = 42) -> Dict[int, Dict[float, float]]:
    """
    Monte Carlo per probabilità di CALO: P(crescita cumulata <= 1/threshold) per la
    stessa percentuale threshold (es. threshold=1.05 -> calo 5% => <= 1/1.05).
    Ritorna dizionario: {h: {threshold: prob_calo}}.
    """
    rng = np.random.default_rng(seed)
    T_max = int(max(horizons))
    shocks = rng.normal(loc=mu, scale=sigma, size=(T_max, int(trials)))
    cum_log = np.cumsum(shocks, axis=0)
    results: Dict[int, Dict[float, float]] = {}
    for h in horizons:
        idx = int(h) - 1
        cum_growth = np.exp(cum_log[idx, :])
        results[int(h)] = {}
        for thr in thresholds:
            decline_level = 1.0 / float(thr)
            results[int(h)][float(thr)] = float(np.mean(cum_growth <= decline_level))
    return results


def simulate_paths(mu: float, sigma: float, n_paths: int, T: int,
                   seed: Optional[int] = 123) -> np.ndarray:
    """
    Simula percorsi e restituisce array (T+1, n_paths) con t=0 valore 1.0.
    """
    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=mu, scale=sigma, size=(int(T), int(n_paths)))
    cum_log = np.cumsum(shocks, axis=0)
    paths = np.exp(cum_log)
    paths = np.vstack([np.ones((1, int(n_paths))), paths])
    return paths


# ---------- Analitico (chiuso) ----------
def analytic_probs(mu: float, sigma: float,
                   horizons: List[int] = HORIZONS,
                   thresholds: List[float] = THRESHOLDS) -> Tuple[Dict[int, Dict[float, float]], Dict[int, Dict[float, float]]]:
    """
    Calcolo analitico (quando disponibile scipy.stats.norm).
    Ritorna due dizionari: (growth_probs, decline_probs).
    growth_probs[h][thr] = P(exp(S_h) >= thr).
    decline_probs[h][thr] = P(exp(S_h) <= 1/thr).
    """
    try:
        from scipy.stats import norm  # type: ignore
    except Exception as e:
        raise RuntimeError("scipy non disponibile per il calcolo analitico.") from e

    growth_res: Dict[int, Dict[float, float]] = {}
    decline_res: Dict[int, Dict[float, float]] = {}
    for h in horizons:
        mean = h * mu
        std = math.sqrt(h) * sigma
        growth_res[int(h)] = {}
        decline_res[int(h)] = {}
        for thr in thresholds:
            thr = float(thr)
            if std <= 0:
                pg = 1.0 if mean >= math.log(thr) else 0.0
                pd = 1.0 if mean <= math.log(1.0 / thr) else 0.0
            else:
                z_g = (math.log(thr) - mean) / std
                pg = float(1.0 - norm.cdf(z_g))
                z_d = (math.log(1.0 / thr) - mean) / std
                pd = float(norm.cdf(z_d))
            growth_res[int(h)][thr] = pg
            decline_res[int(h)][thr] = pd
    return growth_res, decline_res


def results_to_dataframe(results: Dict[int, Dict[float, float]], label_prefix: str = "") -> pd.DataFrame:
    """
    Converte result dict {h: {thr:prob}} in DataFrame con colonne leggibili.
    label_prefix è prefisso per le colonne (es. 'MC ' o 'Analitico ').
    """
    rows = []
    for h in sorted(results.keys()):
        row = {"Giorni": int(h)}
        for thr, p in sorted(results[h].items()):
            col = f"{label_prefix}>= { (thr - 1.0) * 100 :.1f}%"
            row[col] = p
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Giorni").reset_index(drop=True)
    return df


# ---------- Utility per confronto e posizionamento ----------
def build_comparison_table(mc_up: Dict[int, Dict[float, float]],
                           mc_down: Dict[int, Dict[float, float]],
                           mu: float, sigma: float) -> pd.DataFrame:
    """
    Costruisce tabella di confronto per ogni (giorni, soglia):
    Giorni, Soglia, P_up (MC), P_down (MC), Diff, Exp_growth (valore atteso), Status
    Status: "Tilt verso crescita" se P_up > P_down, "Tilt verso calo" se P_down > P_up, "Neutrale" altrimenti.
    Exp_growth = E[exp(S_h)] = exp(h*mu + 0.5*h*sigma^2)
    """
    rows = []
    for h in sorted(mc_up.keys()):
        exp_growth = math.exp(h * mu + 0.5 * (h * sigma * sigma))
        for thr in sorted(mc_up[h].keys()):
            p_up = mc_up[h][thr]
            p_down = mc_down[h].get(thr, 0.0)
            diff = p_up - p_down
            if diff > 0.01:
                status = "Tilt verso crescita"
            elif diff < -0.01:
                status = "Tilt verso calo"
            else:
                status = "Neutrale"
            rows.append({
                "Giorni": int(h),
                "Soglia%": f"{(thr - 1.0) * 100:.1f}%",
                "P_crescita_MC": p_up,
                "P_calo_MC": p_down,
                "Diff (cresc-calo)": diff,
                "Exp_growth (E[prod])": exp_growth,
                "Status": status
            })
    df = pd.DataFrame(rows).sort_values(["Giorni", "Soglia%"]).reset_index(drop=True)
    return df


# ---------- Streamlit UI ----------

def render(*_: Any):
    import streamlit as st
    import matplotlib.pyplot as plt

    st.header("Probabilità di crescita e di calo — Monte Carlo + comparazione")

    with st.expander("Parametri", expanded=True):
        c1, c2 = st.columns(2)
        mu = c1.number_input("mu log-rend. giornaliero", value=0.0005, step=0.0001, format="%.6f")
        sigma = c2.number_input("sigma log-rend. giornaliero", value=0.02, step=0.001, format="%.6f")

        c3, c4 = st.columns(2)
        trials = int(c3.number_input("Simulazioni (Monte Carlo)", value=200_000, step=50_000, min_value=1_000))
        seed = int(c4.number_input("Seed RNG", value=42, step=1))

        c5, c6 = st.columns(2)
        thr_str = c5.text_input("Soglie cumulative (es. 1.025,1.05,1.10)", ",".join([str(x) for x in THRESHOLDS]))
        hor_str = c6.text_input("Orizzonti giorni (es. 30,60,90,120)", ",".join([str(x) for x in HORIZONS]))

        # valore attuale per confronto (es. fattore cumulato attuale rispetto baseline)
        c7, c8 = st.columns(2)
        current_value = c7.number_input("Valore attuale (1.0 = baseline, es. 1.02 = +2%)", value=1.0, step=0.001, format="%.6f")
        show_current_annotations = c8.checkbox("Mostra annotazioni rispetto al valore attuale", value=True)

        try:
            thresholds = [float(x.strip()) for x in thr_str.split(",") if x.strip()]
            horizons = [int(x.strip()) for x in hor_str.split(",") if x.strip()]
        except Exception:
            st.error("Formato soglie/orizzonti non valido.")
            return

    if st.button("Calcola probabilità, confronto e grafico"):
        if sigma <= 0 or not np.isfinite(sigma):
            st.error("sigma deve essere > 0.")
            return

        with st.spinner("Esecuzione Monte Carlo..."):
            mc_up = simulate_probs(mu=mu, sigma=sigma, trials=trials, horizons=horizons, thresholds=thresholds, seed=seed)
            mc_down = simulate_probs_decline(mu=mu, sigma=sigma, trials=trials, horizons=horizons, thresholds=thresholds, seed=seed)

        # Analitico (se possibile)
        try:
            an_up, an_down = analytic_probs(mu=mu, sigma=sigma, horizons=horizons, thresholds=thresholds)
            analytic_ok = True
        except Exception:
            an_up, an_down = {}, {}
            analytic_ok = False

        # Mostra MC growth table
        st.subheader("Monte Carlo — Probabilità di crescita (>= soglia)")
        df_mc_up = results_to_dataframe(mc_up, label_prefix="MC ")
        df_mc_up_disp = df_mc_up.copy()
        for c in df_mc_up_disp.columns:
            if c != "Giorni":
                df_mc_up_disp[c] = df_mc_up_disp[c].map(lambda x: f"{x:.2%}")
        st.dataframe(df_mc_up_disp, use_container_width=True)

        st.subheader("Monte Carlo — Probabilità di calo (<= 1/soglia)")
        df_mc_down = results_to_dataframe(mc_down, label_prefix="MC ")
        df_mc_down_disp = df_mc_down.copy()
        for c in df_mc_down_disp.columns:
            if c != "Giorni":
                df_mc_down_disp[c] = df_mc_down_disp[c].map(lambda x: f"{x:.2%}")
        st.dataframe(df_mc_down_disp, use_container_width=True)

        if analytic_ok:
            st.subheader("Analitico — Probabilità di crescita")
            df_an_up = results_to_dataframe(an_up, label_prefix="AN ")
            df_an_up_disp = df_an_up.copy()
            for c in df_an_up_disp.columns:
                if c != "Giorni":
                    df_an_up_disp[c] = df_an_up_disp[c].map(lambda x: f"{x:.2%}")
            st.dataframe(df_an_up_disp, use_container_width=True)

            st.subheader("Analitico — Probabilità di calo")
            df_an_down = results_to_dataframe(an_down, label_prefix="AN ")
            df_an_down_disp = df_an_down.copy()
            for c in df_an_down_disp.columns:
                if c != "Giorni":
                    df_an_down_disp[c] = df_an_down_disp[c].map(lambda x: f"{x:.2%}")
            st.dataframe(df_an_down_disp, use_container_width=True)

        # Confronto MC crescita vs calo e posizionamento
        st.subheader("Confronto Monte Carlo: crescita vs calo + posizionamento attuale")
        cmp_df = build_comparison_table(mc_up, mc_down, mu, sigma)
        cmp_df_display = cmp_df.copy()
        for col in ["P_crescita_MC", "P_calo_MC", "Diff (cresc-calo)"]:
            if col != "Diff (cresc-calo)":
                cmp_df_display[col] = cmp_df_display[col].map(lambda x: f"{x:.2%}")
            else:
                cmp_df_display[col] = cmp_df_display[col].map(lambda x: f"{float(x):+.2%}")
        cmp_df_display["Exp_growth (E[prod])"] = cmp_df_display["Exp_growth (E[prod])"].map(lambda x: f"{x:.2f}x")
        st.dataframe(cmp_df_display, use_container_width=True)

        # Sintesi testuale del posizionamento corrente
        st.markdown("### Posizionamento sintetico")
        synth_lines: List[str] = []
        for h in sorted(mc_up.keys()):
            exp_growth = math.exp(h * mu + 0.5 * (h * sigma * sigma))
            median_growth = math.exp(h * mu)
            first_thr = sorted(mc_up[h].keys())[0]
            p_up = mc_up[h][first_thr]
            p_down = mc_down[h][first_thr]
            tilt = "Tilt verso crescita" if p_up > p_down else ("Tilt verso calo" if p_down > p_up else "Neutrale")
            synth_lines.append(
                f"- Orizzonte {h} giorni: aspettativa E[crescita] = {exp_growth:.3f}x "
                f"(mediana {median_growth:.3f}x). Primo thr { (first_thr-1.0)*100 :.1f}% -> P_up={p_up:.2%}, P_down={p_down:.2%}. {tilt}."
            )
        for ln in synth_lines:
            st.markdown(ln)

        st.caption("Interpretazione: Exp_growth >1 indica aspettativa positiva media nel periodo. "
                   "Probabilità di crescita vs calo mostrano il bilanciamento del rischio per le soglie selezionate.")

        # --- Tracce percorsi per visuale ---
        T_plot = max(horizons)
        n_paths = min(50, max(20, int(trials // 5000)))
        paths = simulate_paths(mu=mu, sigma=sigma, n_paths=n_paths, T=T_plot, seed=seed + 1)

        t = np.arange(paths.shape[0])
        median_path = np.median(paths, axis=1)
        p5 = np.percentile(paths, 5, axis=1)
        p95 = np.percentile(paths, 95, axis=1)

        fig, ax = plt.subplots(figsize=(9, 4))
        for i in range(n_paths):
            ax.plot(t, paths[:, i], linewidth=0.8, alpha=0.4)
        ax.plot(t, median_path, linewidth=2, label="Mediana")
        ax.fill_between(t, p5, p95, alpha=0.25, label="5°–95°")
        ax.set_xlabel("Giorni")
        ax.set_ylabel("Crescita cumulata (x)")
        ax.set_title(f"Simulazioni Monte Carlo — {n_paths} percorsi fino a {T_plot} giorni")
        ax.grid(True, alpha=0.25)
        ax.legend()
        st.pyplot(fig)

        # ================== NUOVI GRAFICI DI CONFRONTO ==================
        st.subheader("Grafici confronto: Probabilità Crescita vs Calo per soglie e orizzonti")

        # Prepara matrici per plotting
        horizons_sorted = sorted(mc_up.keys())
        thresholds_sorted = sorted(next(iter(mc_up.values())).keys()) if mc_up else sorted(thresholds)

        up_mat = np.zeros((len(horizons_sorted), len(thresholds_sorted)))
        down_mat = np.zeros_like(up_mat)
        for i, h in enumerate(horizons_sorted):
            for j, thr in enumerate(thresholds_sorted):
                up_mat[i, j] = mc_up[h].get(thr, 0.0)
                down_mat[i, j] = mc_down[h].get(thr, 0.0)

        # 1) Grafico a barre raggruppate per orizzonte: per ogni orizzonte un subplot con barre P_up vs P_down per soglia
        n_h = len(horizons_sorted)
        fig1, axs = plt.subplots(n_h, 1, figsize=(10, 3 * n_h), squeeze=False)
        for idx, h in enumerate(horizons_sorted):
            axh = axs[idx, 0]
            x = np.arange(len(thresholds_sorted))
            width = 0.35
            axh.bar(x - width/2, up_mat[idx, :], width, label='P Crescita (MC)', alpha=0.9)
            axh.bar(x + width/2, down_mat[idx, :], width, label='P Calo (MC)', alpha=0.9)
            axh.set_xticks(x)
            axh.set_xticklabels([f"{(thr-1.0)*100:.1f}%" for thr in thresholds_sorted])
            axh.set_ylim(0, 1.0)
            axh.set_ylabel("Probabilità")
            axh.set_title(f"Orizzonte {h} giorni")
            axh.legend()
            axh.grid(axis='y', alpha=0.25)
            # annotazioni rispetto al valore attuale (se richieste)
            if show_current_annotations:
                ann_texts = []
                for j, thr in enumerate(thresholds_sorted):
                    if current_value >= thr:
                        axh.text(j + 0.02, 0.95, "Già > thr", fontsize=8, color='green', ha='center', va='top', transform=axh.get_xaxis_transform())
                    elif current_value <= 1.0 / thr:
                        axh.text(j + 0.02, 0.95, "Già < 1/thr", fontsize=8, color='red', ha='center', va='top', transform=axh.get_xaxis_transform())
        plt.tight_layout()
        st.pyplot(fig1)

        # 2) Heatmap della differenza P_up - P_down (tilt) per horizon x thresholds
        diff_mat = up_mat - down_mat
        fig2, ax2 = plt.subplots(figsize=(10, 3 + 0.6 * len(horizons_sorted)))
        c = ax2.imshow(diff_mat, aspect='auto', cmap='RdYlGn', vmin=-1, vmax=1)
        ax2.set_yticks(np.arange(len(horizons_sorted)))
        ax2.set_yticklabels([str(h) + "d" for h in horizons_sorted])
        ax2.set_xticks(np.arange(len(thresholds_sorted)))
        ax2.set_xticklabels([f"{(thr-1.0)*100:.1f}%" for thr in thresholds_sorted], rotation=45)
        ax2.set_title("Tilt (P_crescita - P_calo) per Orizzonte x Soglia")
        for (i, j), val in np.ndenumerate(diff_mat):
            ax2.text(j, i, f"{val:+.1%}", ha='center', va='center', color='black', fontsize=9)
        fig2.colorbar(c, ax=ax2, fraction=0.046, pad=0.04, label='Diff')
        plt.tight_layout()
        st.pyplot(fig2)

        # 3) Line plot: Exp_growth e Mediana vs Orizzonte, con linea valore attuale orizzonte-indipendente
        fig3, ax3 = plt.subplots(figsize=(9, 4))
        exp_vals = [math.exp(h * mu + 0.5 * (h * sigma * sigma)) for h in horizons_sorted]
        med_vals = [math.exp(h * mu) for h in horizons_sorted]
        ax3.plot(horizons_sorted, exp_vals, marker='o', label='E[crescita] (media)', linewidth=2)
        ax3.plot(horizons_sorted, med_vals, marker='o', label='Mediana', linewidth=2)
        ax3.hlines(current_value, xmin=min(horizons_sorted), xmax=max(horizons_sorted), colors='purple', linestyles='--', label='Valore attuale')
        ax3.set_xlabel("Giorni")
        ax3.set_ylabel("Fattore cumulato (x)")
        ax3.set_title("Aspettativa media e mediana vs orizzonte (con valore attuale)")
        ax3.grid(alpha=0.25)
        ax3.legend()
        st.pyplot(fig3)

        # 4) Tabella di posizionamento rispetto a soglie: mostra se valore attuale già supera thr o 1/thr
        pos_rows = []
        for h in horizons_sorted:
            for thr in thresholds_sorted:
                status = "Neutro"
                if current_value >= thr:
                    status = "Attualmente >= soglia (già superata)"
                elif current_value <= 1.0 / thr:
                    status = "Attualmente <= 1/soglia (già sotto)"
                pos_rows.append({
                    "Giorni": h,
                    "Soglia%": f"{(thr-1.0)*100:.1f}%",
                    "Valore attuale": current_value,
                    "Stato relativo": status,
                    "P_crescita_MC": mc_up[h][thr],
                    "P_calo_MC": mc_down[h][thr],
                })
        pos_df = pd.DataFrame(pos_rows)
        pos_df_display = pos_df.copy()
        pos_df_display["P_crescita_MC"] = pos_df_display["P_crescita_MC"].map(lambda x: f"{x:.2%}")
        pos_df_display["P_calo_MC"] = pos_df_display["P_calo_MC"].map(lambda x: f"{x:.2%}")
        st.markdown("### Stato attuale rispetto alle soglie")
        st.dataframe(pos_df_display, use_container_width=True)

# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description="Monte Carlo per probabilità di crescita e calo cumulati.")
    p.add_argument("--mu", type=float, default=0.0005, help="Media log-rendimenti giornalieri. Default 0.0005")
    p.add_argument("--sigma", type=float, default=0.02, help="Dev. std log-rendimenti giornalieri. Default 0.02")
    p.add_argument("--trials", type=int, default=200_000, help="Numero simulazioni.")
    p.add_argument("--seed", type=int, default=42, help="Seed RNG.")
    p.add_argument("--thresholds", type=str, default="1.025,1.05,1.10", help="Soglie cumulative (es. 1.05 per +5%).")
    p.add_argument("--horizons", type=str, default="30,60,90,120", help="Orizzonti giorni.")
    p.add_argument("--analytic", action="store_true", help="Mostra anche le probabilità analitiche (richiede scipy).")
    args = p.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    if args.sigma <= 0 or not np.isfinite(args.sigma):
        raise SystemExit("sigma deve essere > 0.")

    mc_up = simulate_probs(mu=args.mu, sigma=args.sigma, trials=args.trials, horizons=horizons, thresholds=thresholds, seed=args.seed)
    mc_down = simulate_probs_decline(mu=args.mu, sigma=args.sigma, trials=args.trials, horizons=horizons, thresholds=thresholds, seed=args.seed)

    print("Monte Carlo — Probabilità di crescita (>= soglia):")
    df_mc_up = results_to_dataframe(mc_up, label_prefix="MC ")
    pd.set_option("display.float_format", lambda x: f"{x:.2%}")
    print(df_mc_up.to_string(index=False))

    print("\nMonte Carlo — Probabilità di calo (<= 1/soglia):")
    df_mc_down = results_to_dataframe(mc_down, label_prefix="MC ")
    print(df_mc_down.to_string(index=False))

    if args.analytic:
        try:
            an_up, an_down = analytic_probs(mu=args.mu, sigma=args.sigma, horizons=horizons, thresholds=thresholds)
            print("\nAnalitico — Probabilità di crescita (somma di normali):")
            df_an_up = results_to_dataframe(an_up, label_prefix="AN ")
            print(df_an_up.to_string(index=False))
            print("\nAnalitico — Probabilità di calo (somma di normali):")
            df_an_down = results_to_dataframe(an_down, label_prefix="AN ")
            print(df_an_down.to_string(index=False))
        except Exception as e:
            print("\nAnalitico non disponibile (scipy richiesta). Errore:", e)

    # confronto sintetico e posizionamento
    cmp_df = build_comparison_table(mc_up, mc_down, args.mu, args.sigma)
    print("\nConfronto MC: crescita vs calo e posizionamento (prime righe):")
    pd.set_option("display.float_format", lambda x: f"{x:.2%}")
    cmp_print = cmp_df.copy()
    cmp_print["Exp_growth (E[prod])"] = cmp_print["Exp_growth (E[prod])"].map(lambda x: f"{x:.2f}x")
    print(cmp_print.head(20).to_string(index=False))

    print("\nSintesi posizionamento per orizzonte (aspettativa E[crescita]):")
    for h in sorted(mc_up.keys()):
        exp_growth = math.exp(h * args.mu + 0.5 * (h * args.sigma * args.sigma))
        median_growth = math.exp(h * args.mu)
        first_thr = sorted(mc_up[h].keys())[0]
        p_up = mc_up[h][first_thr]
        p_down = mc_down[h][first_thr]
        tilt = "Tilt verso crescita" if p_up > p_down else ("Tilt verso calo" if p_down > p_up else "Neutrale")
        print(f"- {h} giorni: E[crescita]={exp_growth:.3f}x (median {median_growth:.3f}x). "
              f"primo thr {(first_thr-1.0)*100:.1f}% -> P_up={p_up:.2%}, P_down={p_down:.2%}. {tilt}.")


if __name__ == "__main__":
    main()
