"""
Questão 3 — Identificação de Prejuízo por Produto
===================================================
Câmbio: cotação PTAX de venda do Banco Central do Brasil (API oficial)
Fonte:  https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/CotacaoDolarPeriodo
"""

import pandas as pd
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import date
from dateutil.relativedelta import relativedelta

# ═════════════════════════════════════════════════════════════════════════════
# PARTE 0 — CARREGAMENTO DOS DADOS
# ═════════════════════════════════════════════════════════════════════════════

df_vendas = pd.read_csv('datasets/vendas_2023_2024.csv')

def parse_date(d):
    for fmt in ['%Y-%m-%d', '%d-%m-%Y']:
        try:
            return pd.to_datetime(d, format=fmt)
        except ValueError:
            pass
    return pd.NaT

df_vendas['date'] = df_vendas['sale_date'].apply(parse_date)

with open('datasets/custos_importacao.json', encoding='utf-8') as f:
    custos_raw = json.load(f)

rows = []
for product in custos_raw:
    for entry in product['historic_data']:
        rows.append({
            'product_id':    product['product_id'],
            'product_name':  product['product_name'],
            'start_date':    pd.to_datetime(entry['start_date'], format='%d/%m/%Y'),
            'usd_cost_unit': entry['usd_price'],
        })
df_custos = pd.DataFrame(rows).sort_values(['product_id', 'start_date'])

# ═════════════════════════════════════════════════════════════════════════════
# PARTE 1.A — CÂMBIO PTAX REAL (API BCB — busca mês a mês)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_ptax_month(year: int, month: int) -> pd.DataFrame:
    """
    Busca PTAX de venda para um único mês.
    A API do BCB exige o formato MM-DD-YYYY e não aceita intervalos longos,
    por isso buscamos mês a mês.
    """
    start = date(year, month, 1)
    end   = (start + relativedelta(months=1)) - relativedelta(days=1)

    url = (
        "https://olinda.bcb.gov.br/olinda/servico/PTAX/versao/v1/odata/"
        f"CotacaoDolarPeriodo(dataInicial=@i,dataFinal=@f)"
        f"?@i='{start.strftime('%m-%d-%Y')}'"
        f"&@f='{end.strftime('%m-%d-%Y')}'"
        f"&$format=json&$select=cotacaoVenda,dataHoraCotacao"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json().get('value', [])
    if not data:
        return pd.DataFrame(columns=['date', 'ptax_venda'])

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['dataHoraCotacao']).dt.normalize()
    df = (df.groupby('date')['cotacaoVenda']
            .mean()
            .reset_index()
            .rename(columns={'cotacaoVenda': 'ptax_venda'}))
    return df


def fetch_ptax_range(start_year: int, start_month: int,
                     end_year: int,   end_month: int) -> pd.DataFrame:
    """Busca PTAX mês a mês e concatena."""
    chunks  = []
    current = date(start_year, start_month, 1)
    end_dt  = date(end_year,   end_month,   1)

    while current <= end_dt:
        print(f"   Buscando PTAX {current.strftime('%m/%Y')}...", end='\r')
        try:
            chunk = fetch_ptax_month(current.year, current.month)
            chunks.append(chunk)
        except Exception as e:
            print(f"\n   ⚠ Erro em {current.strftime('%m/%Y')}: {e}")
        current += relativedelta(months=1)

    print()
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


print("▶ Buscando cotações PTAX do Banco Central (2023–2024)...")
df_ptax = fetch_ptax_range(2023, 1, 2024, 12)
print(f"   {len(df_ptax)} dias úteis com cotação carregados")
print(f"   Câmbio mín: {df_ptax['ptax_venda'].min():.4f} | máx: {df_ptax['ptax_venda'].max():.4f}")

# Dias sem cotação (fins de semana / feriados) → forward fill com último dia útil
all_days        = pd.DataFrame({'date': pd.date_range('2023-01-01', '2024-12-31')})
df_ptax         = all_days.merge(df_ptax, on='date', how='left')
df_ptax['ptax_venda'] = df_ptax['ptax_venda'].ffill()

# ═════════════════════════════════════════════════════════════════════════════
# PARTE 1.B — CUSTO VIGENTE NA DATA DA VENDA (merge_asof)
# ═════════════════════════════════════════════════════════════════════════════
print("▶ Cruzando custo histórico com data da venda...")

partes = []
for pid, grp_sales in df_vendas.groupby('id_product'):
    custos_prod = df_custos[df_custos['product_id'] == pid].sort_values('start_date')
    if custos_prod.empty:
        grp = grp_sales.copy()
        grp['usd_cost_unit'] = np.nan
        partes.append(grp)
        continue
    merged = pd.merge_asof(
        grp_sales.sort_values('date'),
        custos_prod[['start_date', 'usd_cost_unit']],
        left_on='date',
        right_on='start_date',
        direction='backward'
    )
    partes.append(merged)

df = pd.concat(partes).reset_index(drop=True)

# ═════════════════════════════════════════════════════════════════════════════
# PARTE 1.C — CÁLCULO DO CUSTO BRL E LUCRO/PREJUÍZO
# ═════════════════════════════════════════════════════════════════════════════

df = df.merge(df_ptax, on='date', how='left')

# custo total BRL = custo_usd_unitário × câmbio_do_dia × quantidade
df['custo_brl_total'] = df['usd_cost_unit'] * df['ptax_venda'] * df['qtd']
df['lucro']           = df['total'] - df['custo_brl_total']

print(f"   Transações com prejuízo: {(df['lucro'] < 0).sum()} ({(df['lucro'] < 0).mean()*100:.1f}%)")

# ═════════════════════════════════════════════════════════════════════════════
# PARTE 1.D — AGREGAÇÃO POR PRODUTO
# ═════════════════════════════════════════════════════════════════════════════
print("▶ Agregando por produto...")

nomes_dict = {c['product_id']: c['product_name'] for c in custos_raw}

receita_total  = df.groupby('id_product')['total'].sum().rename('receita_total')
prejuizo_total = (df[df['lucro'] < 0]
                  .groupby('id_product')['lucro']
                  .sum()
                  .rename('prejuizo_total'))

agg = pd.concat([receita_total, prejuizo_total], axis=1).fillna(0).reset_index()
agg['pct_perda']    = (agg['prejuizo_total'].abs() / agg['receita_total'] * 100).round(2)
agg['product_name'] = agg['id_product'].map(nomes_dict)

agg_prejuizo = agg[agg['prejuizo_total'] < 0].sort_values('prejuizo_total')

print(f"\n   Produtos com prejuízo: {len(agg_prejuizo)}")
print(agg_prejuizo[['product_name', 'receita_total', 'prejuizo_total', 'pct_perda']].head(10).to_string(index=False))

# ═════════════════════════════════════════════════════════════════════════════
# PARTE 2 — GRÁFICO
# ═════════════════════════════════════════════════════════════════════════════
print("\n▶ Gerando gráfico...")

top15 = agg_prejuizo.head(15).copy()
top15['prejuizo_M'] = top15['prejuizo_total'] / 1e6
top15['label'] = (top15['product_name']
                  .str.replace('Motor de Popa', 'M. Popa')
                  .str.replace('Motor Diesel',  'M. Diesel')
                  .str.replace('Motor Elétrico', 'M. Elét.'))

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#0a1628')
ax.set_facecolor('#0d1f38')

colors = ['#e8394a' if i < 3 else '#c02233' for i in range(len(top15))]
bars   = ax.barh(top15['label'], top15['prejuizo_M'],
                 color=colors, edgecolor='none', height=0.65)

for bar, val in zip(bars, top15['prejuizo_M']):
    ax.text(val - 0.1, bar.get_y() + bar.get_height() / 2,
            f'R$ {val:.1f}M', va='center', ha='right',
            color='white', fontsize=9, fontweight='bold')

ax.set_xlabel('Prejuízo Total (R$ Milhões)', color='#8899aa', fontsize=11)
ax.set_title('Top 15 Produtos com Maior Prejuízo Acumulado\n2023–2024 · Câmbio PTAX Banco Central',
             color='#f0f4f8', fontsize=13, fontweight='bold', pad=15)
ax.tick_params(colors='#8899aa')
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'R${x:.0f}M'))
for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(axis='x', color='#1e3050', linewidth=0.8)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('grafico_prejuizo_produtos.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.show()
print("   Gráfico salvo: grafico_prejuizo_produtos.png")

# ═════════════════════════════════════════════════════════════════════════════
# PARTE 3 — ANÁLISE OBJETIVA
# ═════════════════════════════════════════════════════════════════════════════

maior_abs = agg_prejuizo.iloc[0]
maior_pct = agg_prejuizo.loc[agg_prejuizo['pct_perda'].idxmax()]

print("\n" + "═"*60)
print("PARTE 3 — ANÁLISE OBJETIVA")
print("═"*60)
print(f"\n→ Produto com maior prejuízo absoluto:")
print(f"  {maior_abs['product_name']}")
print(f"  Prejuízo: R$ {maior_abs['prejuizo_total']:,.2f}")
print(f"  % de perda: {maior_abs['pct_perda']:.2f}%")

print(f"\n→ Produto com maior % de perda:")
print(f"  {maior_pct['product_name']}")
print(f"  % de perda: {maior_pct['pct_perda']:.2f}%")
print(f"  Prejuízo absoluto: R$ {maior_pct['prejuizo_total']:,.2f}")

mesmo = maior_abs['id_product'] == maior_pct['id_product']
print(f"\n→ São o mesmo produto? {'SIM' if mesmo else 'NÃO'}")