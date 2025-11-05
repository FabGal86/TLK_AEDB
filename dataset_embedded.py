from pathlib import Path
import io, pandas as pd

# Cerca CSV in questi posti (prima preferenza)
_SEARCH_PATHS = [
    Path(__file__).parent / "data",   # ./data/
    Path(__file__).parent,            # repo root
    Path("/app/data"),
    Path("/mnt/data"),
]

def _try_read_bytes(b: bytes):
    for sep in [';', ',', '\t', '|']:
        try:
            df = pd.read_csv(io.BytesIO(b), sep=sep, engine='python', encoding='utf-8-sig')
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    # fallback permissivo: prova comunque a leggere
    return pd.read_csv(io.BytesIO(b), engine='python', encoding='utf-8-sig')

def list_tables():
    names = set()
    # file CSV nei percorsi noti
    for base in _SEARCH_PATHS:
        try:
            for p in base.glob("*.csv"):
                names.add(p.stem)
        except Exception:
            continue
    return sorted(names)

def load_table(name: str):
    # 1) cerca file CSV su disco
    for base in _SEARCH_PATHS:
        p = base / f"{name}.csv"
        try:
            if p.exists():
                return _try_read_bytes(p.read_bytes())
        except Exception:
            continue
    # 2) nessun CSV trovato -> errore informativo
    raise KeyError(f"Table '{name}' non trovata. Controlla che { [str(p) for p in _SEARCH_PATHS] } contengano {name}.csv")
