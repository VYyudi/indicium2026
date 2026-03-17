import pandas as pd
import unicodedata

df = pd.read_csv('datasets/produtos_raw.csv')

# ── PARTE 1 — Padronização de categorias ─────────────────────────────────────
def normalize_category(s: str) -> str:
    s_norm = "".join(
        c for c in unicodedata.normalize("NFD", s.strip().lower().replace(" ", ""))
        if unicodedata.category(c) != "Mn"
    )
    if "eletron" in s_norm or "eletrun" in s_norm:
        return "eletrônicos"
    elif "propul" in s_norm or "propuc" in s_norm or s_norm == "prop":
        return "propulsão"
    elif "ancor" in s_norm or "encor" in s_norm:
        return "ancoragem"
    return s.strip().lower()

df["actual_category"] = df["actual_category"].apply(normalize_category)

# ── PARTE 2 — Conversão de valores para numérico ─────────────────────────────
df["price"] = (
    df["price"]
    .str.replace("R$ ", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

# ── PARTE 3 — Remoção de duplicatas ──────────────────────────────────────────
before = len(df)
df = df.drop_duplicates()
after = len(df)

print(f"Categorias únicas: {df['actual_category'].unique()}")
print(f"Tipo da coluna price: {df['price'].dtype}")
print(f"Duplicatas removidas: {before - after}")
print(df.head())
