# make_embedded_from_desktop.py  (salva così)
from pathlib import Path
import base64, gzip

CSV_DIR = Path(r"C:\Users\HP\Desktop\DATABASE_TLK")
OUT = Path(__file__).parent / "dataset_embedded.py"

if not CSV_DIR.exists():
    raise SystemExit(f"Cartella CSV non trovata: {CSV_DIR}")

csvs = sorted(CSV_DIR.glob("*.csv"))
if not csvs:
    raise SystemExit(f"Nessun CSV in {CSV_DIR}")

def compress_b64(b: bytes) -> str:
    return base64.b64encode(gzip.compress(b)).decode("ascii")

lines = []
lines.append("# Auto-generated dataset_embedded.py (CSV compressed)\n")
lines.append("import base64, gzip, io, pandas as pd\n\n")
lines.append("__EMBEDDED__ = {\n")

for p in csvs:
    name = p.stem
    print("Processing:", p.name)
    b = p.read_bytes()
    b64 = compress_b64(b)
    # scriviamo la mappa (attenzione: file potenzialmente grande)
    lines.append(f"    {name!r}: {b64!r},\n")

lines.append("}\n\n")
lines.append("def _decode(b64str):\n")
lines.append("    gz = base64.b64decode(b64str.encode('ascii'))\n")
lines.append("    return gzip.decompress(gz)\n\n")
lines.append("def list_tables():\n")
lines.append("    return list(__EMBEDDED__.keys())\n\n")
lines.append("def load_table(name):\n")
lines.append("    if name not in __EMBEDDED__:\n")
lines.append("        raise KeyError(name)\n")
lines.append("    b = _decode(__EMBEDDED__[name])\n")
lines.append("    return pd.read_csv(io.BytesIO(b))\n\n")
lines.append("def load_all():\n")
lines.append("    return {n: load_table(n) for n in list_tables()}\n")

OUT.write_text(''.join(lines), encoding='utf-8')
print("Creato:", OUT.resolve())
print("Tabelle embed:", [p.stem for p in csvs])
