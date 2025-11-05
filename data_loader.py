# data_loader.py
import io
import os
from pathlib import Path
import pandas as pd

# Percorsi dove cerchiamo file CSV nel repo / deploy
SEARCH_PATHS = [
    Path("data"),              # ./data/<table>.csv
    Path("."),                 # ./<table>.csv
    Path("/app/data"),         # common deploy path
    Path("/mnt/data"),         # ambiente di debug qui
]

def _try_read_bytes(b):
    """Prova separatori comuni e utf-8-sig, ritorna DataFrame o None."""
    for sep in [',', ';', '\t', '|']:
        try:
            df = pd.read_csv(io.BytesIO(b), sep=sep, engine='python', encoding='utf-8-sig')
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
    # fallback permissivo
    try:
        return pd.read_csv(io.BytesIO(b), encoding='utf-8-sig')
    except Exception:
        return None

def load_table(name):
    """Carica tabella cercando (in ordine):
       1) dataset_embedded.load_table (se esiste)
       2) ./data/<name>.csv (o .csv.gz)
       3) altri percorsi listati in SEARCH_PATHS
       Altrimenti solleva FileNotFoundError con dettagli.
    """
    # 1) prova dataset_embedded
    try:
        import dataset_embedded as de  # se non esiste solleva ImportError
        try:
            return de.load_table(name)
        except KeyError:
            # modulo esiste ma non contiene la tabella: continuiamo con altri fallback
            pass
        except Exception:
            # se il loader embedded fallisce, continuiamo con fallback
            pass
    except Exception:
        pass

    # 2) prova file CSV nei percorsi noti
    tried = []
    for base in SEARCH_PATHS:
        base = Path(base)
        for ext in (".csv", ".csv.gz", ".txt"):
            p = base / f"{name}{ext}"
            tried.append(str(p))
            if p.exists():
                # se è gz mano a pd.read_csv lo gestisce automaticamente
                try:
                    return pd.read_csv(p, engine='python', encoding='utf-8-sig', sep=None)
                except Exception:
                    # fallback a lettura raw con detection di separatore
                    try:
                        b = p.read_bytes()
                        df = _try_read_bytes(b)
                        if df is not None:
                            return df
                    except Exception:
                        continue

    # 3) nessuna sorgente trovata: errore informativo
    raise FileNotFoundError(
        f"load_table('{name}') failed. Nessun dataset embedded trovato e nessun file CSV tra i percorsi: {SEARCH_PATHS}. "
        f"Ho provato questi path: {tried}. Azioni possibili:\n"
        "- aggiungi i CSV in ./data/ o nella root del repo e committa nel repo (oppure usa Git LFS per file grandi)\n"
        "- genera un dataset_embedded.py (usa make_embedded_from_desktop.py se lo hai)\n"
        "- ospita i CSV in uno storage pubblico e modifica il codice per scaricarli al runtime."
    )
