"""
LH Nautical — Pipeline de Dados
================================
Executa toda a limpeza, tratamento e análise dos datasets brutos
e exporta um único arquivo `data.json` para ser lido pelo dashboard HTML.

Estrutura de arquivos esperada:
    datasets/
        produtos_raw.csv
        vendas_2023_2024.csv
        clientes_crm.json
        custos_importacao.json
    pipeline.py
    dashboard.html

Uso:
    python pipeline.py
    → gera data.json na mesma pasta
"""

import pandas as pd
import numpy as np
import json
import unicodedata
from pathlib import Path

# ── CONFIGURAÇÕES ─────────────────────────────────────────────────────────────
DATA_DIR    = Path("datasets")
OUTPUT_FILE = Path("data.json")
USD_TO_BRL  = 5.15   # taxa média representativa 2023-2024 mude conforme necessario

# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — LIMPEZA DE PRODUTOS
# ═════════════════════════════════════════════════════════════════════════════
print("▶ Carregando e limpando produtos...")

df_prod = pd.read_csv(DATA_DIR / "produtos_raw.csv")

# Preço: remove "R$ " e converte para float
df_prod["price_clean"] = (
    df_prod["price"]
    .str.replace("R$ ", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

def normalize_category(s: str) -> str:
    """Normaliza as 35+ variações de categoria para 3 valores canônicos."""
    s_norm = "".join(
        c for c in unicodedata.normalize("NFD", s.strip().lower().replace(" ", ""))
        if unicodedata.category(c) != "Mn"
    )
    if "eletron" in s_norm or "eletrun" in s_norm:
        return "Eletrônicos"
    elif "propul" in s_norm or "propuc" in s_norm or s_norm == "prop":
        return "Propulsão"
    elif "ancor" in s_norm or "encor" in s_norm:
        return "Ancoragem"
    return s.strip().capitalize()

df_prod["category"] = df_prod["actual_category"].apply(normalize_category)

df_prod_clean = df_prod.rename(columns={
    "code":        "product_id",
    "name":        "product_name",
    "price_clean": "sale_price_brl",
})[["product_id", "product_name", "sale_price_brl", "category"]]

print(f"   {len(df_prod_clean)} produtos | categorias: {df_prod_clean['category'].value_counts().to_dict()}")

# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — LIMPEZA DE VENDAS
# ═════════════════════════════════════════════════════════════════════════════
print("▶ Carregando e limpando vendas...")

df_v = pd.read_csv(DATA_DIR / "vendas_2023_2024.csv")

def parse_date(d: str) -> pd.Timestamp:
    """Normaliza os dois formatos de data presentes no dataset."""
    for fmt in ["%Y-%m-%d", "%d-%m-%Y"]:
        try:
            return pd.to_datetime(d, format=fmt)
        except ValueError:
            pass
    return pd.NaT

df_v["date"] = df_v["sale_date"].apply(parse_date)
df_v = df_v.drop(columns=["sale_date"])

print(f"   {len(df_v)} vendas | datas nulas: {df_v['date'].isna().sum()}")
print(f"   Período: {df_v['date'].min().date()} → {df_v['date'].max().date()}")

# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 3 — LIMPEZA DE CLIENTES
# ═════════════════════════════════════════════════════════════════════════════
print("▶ Carregando e limpando clientes...")

with open(DATA_DIR / "clientes_crm.json", encoding="utf-8") as f:
    clientes_raw = json.load(f)

df_cl = pd.DataFrame(clientes_raw).rename(columns={
    "code":      "client_id",
    "full_name": "name",
})

def fix_location(loc: str) -> str:
    """Corrige a inversão Cidade/UF em alguns registros."""
    parts = [p.strip() for p in loc.split(",")]
    if len(parts) == 2 and len(parts[0]) == 2 and parts[0].isupper():
        return f"{parts[1]}, {parts[0]}"
    return loc

df_cl["location"] = df_cl["location"].apply(fix_location)

print(f"   {len(df_cl)} clientes")

# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 4 — CUSTOS DE IMPORTAÇÃO (flatten + merge_asof)
# ═════════════════════════════════════════════════════════════════════════════
print("▶ Processando custos de importação...")

with open(DATA_DIR / "custos_importacao.json", encoding="utf-8") as f:
    custos_raw = json.load(f)

# Transforma o histórico aninhado em linhas planas
rows = []
for c in custos_raw:
    for h in c["historic_data"]:
        rows.append({
            "product_id": c["product_id"],
            "start_date": pd.to_datetime(h["start_date"], format="%d/%m/%Y"),
            "usd_cost":   h["usd_price"],
        })

df_cu = pd.DataFrame(rows).sort_values(["product_id", "start_date"])

# Para cada venda, busca o custo USD vigente na data (sem vazar dados futuros)
partes = []
for pid, grp_sales in df_v.groupby("id_product"):
    custos_prod = df_cu[df_cu["product_id"] == pid].sort_values("start_date")
    if custos_prod.empty:
        grp_sales = grp_sales.copy()
        grp_sales["usd_cost"] = np.nan
        partes.append(grp_sales)
        continue
    merged = pd.merge_asof(
        grp_sales.sort_values("date"),
        custos_prod[["start_date", "usd_cost"]],
        left_on="date",
        right_on="start_date",
        direction="backward",
    )
    partes.append(merged)

df = pd.concat(partes).reset_index(drop=True)

# Calcula custo em BRL e lucro
df["cost_brl_total"] = df["usd_cost"] * USD_TO_BRL * df["qtd"]
df["profit"]         = df["total"] - df["cost_brl_total"]
df["margin_pct"]     = (df["profit"] / df["total"] * 100).round(2)

# Junta nome do cliente e categoria do produto
df = df.merge(df_cl[["client_id", "name"]], left_on="id_client", right_on="client_id", how="left")
df = df.merge(df_prod_clean[["product_id", "product_name", "category"]], left_on="id_product", right_on="product_id", how="left")
df["category"] = df["category"].fillna("Propulsão")  # fallback para 7 produtos sem match

print(f"   Dataset enriquecido: {len(df)} linhas")
print(f"   Receita total: R$ {df['total'].sum():,.0f}")
print(f"   Lucro líquido: R$ {df['profit'].sum():,.0f}")

# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 5 — ANÁLISES E AGREGAÇÕES
# ═════════════════════════════════════════════════════════════════════════════
print("▶ Gerando análises...")

# ── KPIs gerais ───────────────────────────────────────────────────────────────
kpis = {
    "total_records":    int(len(df)),
    "unique_clients":   int(df["id_client"].nunique()),
    "unique_products":  int(df["id_product"].nunique()),
    "total_revenue":    round(df["total"].sum(), 2),
    "total_profit":     round(df["profit"].sum(), 2),
    "avg_ticket":       round(df["total"].mean(), 2),
    "avg_margin":       round(df["margin_pct"].mean(), 2),
    "loss_sales_pct":   round((df["profit"] < 0).mean() * 100, 1),
    "period_start":     df["date"].min().strftime("%Y-%m-%d"),
    "period_end":       df["date"].max().strftime("%Y-%m-%d"),
}

# ── Receita mensal ────────────────────────────────────────────────────────────
monthly = (
    df.groupby(df["date"].dt.to_period("M"))
    .agg(revenue=("total", "sum"), profit=("profit", "sum"), orders=("id", "count"))
    .reset_index()
)
monthly["month"] = monthly["date"].astype(str)
monthly_data = monthly[["month", "revenue", "profit", "orders"]].round(2).to_dict(orient="records")

# ── Vendas médias por dia da semana (incluindo dias sem venda = R$0) ──────────
all_days   = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
daily_rev  = df.groupby("date")["total"].sum().reindex(all_days, fill_value=0)
daily_df   = daily_rev.reset_index()
daily_df.columns = ["date", "revenue"]
daily_df["weekday_num"] = daily_df["date"].dt.dayofweek

day_names_pt = {0: "Segunda", 1: "Terça", 2: "Quarta", 3: "Quinta", 4: "Sexta", 5: "Sábado", 6: "Domingo"}
weekday_avg = (
    daily_df.groupby("weekday_num")["revenue"]
    .mean()
    .round(2)
    .reset_index()
)
weekday_avg["weekday"] = weekday_avg["weekday_num"].map(day_names_pt)
weekday_data = weekday_avg[["weekday", "revenue"]].to_dict(orient="records")

# ── Top 15 produtos com maior prejuízo (Q4) ───────────────────────────────────
loss_products = (
    df[df["profit"] < 0]
    .groupby(["id_product", "product_name", "category"])
    .agg(total_loss=("profit", "sum"), n_sales=("id", "count"))
    .sort_values("total_loss")
    .head(15)
    .reset_index()
)
loss_products_data = loss_products[["product_name", "category", "total_loss", "n_sales"]].round(2).to_dict(orient="records")

# ── Top 15 clientes por lucro acumulado (Q5) ──────────────────────────────────
# Mix de categorias por cliente (% da receita)
cat_mix = (
    df.groupby(["id_client", "category"])["total"]
    .sum()
    .unstack(fill_value=0)
    .apply(lambda row: row / row.sum() * 100, axis=1)
    .round(1)
    .reset_index()
)

client_profit = (
    df.groupby(["id_client", "name"])
    .agg(
        total_revenue=("total", "sum"),
        total_profit=("profit", "sum"),
        num_orders=("id", "count"),
        avg_margin=("margin_pct", "mean"),
    )
    .sort_values("total_profit", ascending=False)
    .head(15)
    .reset_index()
)
client_profit = client_profit.merge(cat_mix, on="id_client", how="left")

# Garante as colunas de categoria mesmo se alguma não existir
for cat in ["Eletrônicos", "Propulsão", "Ancoragem"]:
    if cat not in client_profit.columns:
        client_profit[cat] = 0.0

clients_data = (
    client_profit[["name", "total_revenue", "total_profit", "num_orders", "avg_margin", "Eletrônicos", "Propulsão", "Ancoragem"]]
    .round(2)
    .to_dict(orient="records")
)

# ── Performance por categoria ─────────────────────────────────────────────────
category_data = (
    df.groupby("category")
    .agg(revenue=("total", "sum"), profit=("profit", "sum"), orders=("id", "count"))
    .round(2)
    .reset_index()
    .to_dict(orient="records")
)

# ── Top 10 produtos lucrativos (recomendações) ────────────────────────────────
top_profitable = (
    df.groupby(["id_product", "product_name", "category"])
    .agg(total_profit=("profit", "sum"), n_sales=("id", "count"))
    .sort_values("total_profit", ascending=False)
    .head(10)
    .reset_index()
)
top_profitable_data = top_profitable[["product_name", "category", "total_profit", "n_sales"]].round(2).to_dict(orient="records")

# ── Sazonalidade (índice por mês) ─────────────────────────────────────────────
monthly["month_num"] = monthly["date"].dt.month
season = (
    monthly.groupby("month_num")["revenue"]
    .mean()
    .reset_index()
)
overall_avg = season["revenue"].mean()
season["index"] = ((season["revenue"] - overall_avg) / overall_avg * 100).round(1)
month_names_pt = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
season["month_name"] = season["month_num"].map(month_names_pt)
seasonality_data = season[["month_name", "index"]].to_dict(orient="records")

# ── Previsão simples (regressão linear) ───────────────────────────────────────
from sklearn.linear_model import LinearRegression

X = np.arange(len(monthly)).reshape(-1, 1)
y = monthly["revenue"].values
model = LinearRegression().fit(X, y)

forecast_months = ["2025-01", "2025-02", "2025-03"]
forecast_values = model.predict([[len(monthly)], [len(monthly)+1], [len(monthly)+2]])
forecast_data = [
    {"month": m, "revenue": round(float(v), 2)}
    for m, v in zip(forecast_months, forecast_values)
]

# ═════════════════════════════════════════════════════════════════════════════
# ETAPA 6 — EXPORTA data.json
# ═════════════════════════════════════════════════════════════════════════════
output = {
    "kpis":             kpis,
    "monthly":          monthly_data,
    "weekday":          weekday_data,
    "loss_products":    loss_products_data,
    "clients":          clients_data,
    "categories":       category_data,
    "top_profitable":   top_profitable_data,
    "seasonality":      seasonality_data,
    "forecast":         forecast_data,
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ data.json exportado com sucesso → {OUTPUT_FILE.resolve()}")
print(f"   Chaves: {list(output.keys())}")