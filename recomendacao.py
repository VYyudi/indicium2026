"""
Sistema de Recomendação — Similaridade de Cosseno
===================================================
Produto de referência: GPS Garmin Vortex Maré Drift (id_product = 27)
Lógica: Colaborativa baseada em itens (item-based collaborative filtering)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# 1 — CARREGAMENTO DOS DADOS

df_vendas = pd.read_csv('datasets/vendas_2023_2024.csv')
df_prod   = pd.read_csv('datasets/produtos_raw.csv')

def parse_date(d):
    for fmt in ['%Y-%m-%d', '%d-%m-%Y']:
        try:
            return pd.to_datetime(d, format=fmt)
        except ValueError:
            pass
    return pd.NaT

df_vendas['date'] = df_vendas['sale_date'].apply(parse_date)

# Mapa id → nome do produto
produto_nome = df_prod.set_index('code')['name'].to_dict()

PRODUTO_REF_ID   = 27
PRODUTO_REF_NOME = produto_nome[PRODUTO_REF_ID]
print(f"Produto de referência: [{PRODUTO_REF_ID}] {PRODUTO_REF_NOME}")


# 2 — MATRIZ DE INTERAÇÃO USUÁRIO × PRODUTO (presença/ausência)

# Linhas  : id_client
# Colunas : id_product
# Valor   : 1 se o cliente comprou ao menos uma vez | 0 caso contrário

matriz = (
    df_vendas
    .groupby(['id_client', 'id_product'])['id']  # agrupa por par cliente-produto
    .count()                                      # conta transações (só precisa saber se > 0)
    .gt(0)                                        # vira tudo para True/False
    .astype(int)                                  # True = 1, False = 0
    .unstack(fill_value=0)                        # pivota: clientes nas linhas, produtos nas colunas
)

print(f"\nMatriz Usuário × Produto: {matriz.shape[0]} clientes × {matriz.shape[1]} produtos")
print(matriz.iloc[:5, :8])  # preview


# 3 — SIMILARIDADE DE COSSENO ENTRE PRODUTOS

# Transpõe a matriz para ficar Produto × Cliente
# Cada produto vira um vetor de 0s e 1s indicando quais clientes o compraram
# cosine_similarity compara todos os vetores entre si → matriz produto × produto

matriz_produto = matriz.T  # Produto × Cliente

sim_matrix = cosine_similarity(matriz_produto)

df_sim = pd.DataFrame(
    sim_matrix,
    index=matriz_produto.index,
    columns=matriz_produto.index
)

print(f"\nMatriz de Similaridade: {df_sim.shape[0]} × {df_sim.shape[1]} produtos")


# 4 — RANKING DOS 5 PRODUTOS MAIS SIMILARES AO GPS GARMIN VORTEX MARÉ DRIFT

# Linha do produto de referência — similaridade com todos os outros
similaridades = df_sim[PRODUTO_REF_ID].drop(PRODUTO_REF_ID)  # remove ele mesmo

ranking = (
    similaridades
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
    .rename(columns={'id_product': 'id_produto', PRODUTO_REF_ID: 'similaridade'})
)

ranking['produto_nome']  = ranking['id_produto'].map(produto_nome)
ranking['similaridade']  = ranking['similaridade'].round(4)
ranking.index            = range(1, 6)
ranking.index.name       = 'ranking'

print(f"\n=== TOP 5 RECOMENDAÇÕES PARA ===")
print(f"    '{PRODUTO_REF_NOME}'\n")
print(ranking[['id_produto', 'produto_nome', 'similaridade']].to_string())

# 5 — CONTEXTO: clientes que compraram o GPS (para validar)

clientes_gps = set(df_vendas[df_vendas['id_product'] == PRODUTO_REF_ID]['id_client'])
print(f"\nClientes que compraram o GPS: {len(clientes_gps)} → {sorted(clientes_gps)}")

print("\nTaxa de co-compra dos produtos recomendados:")
for _, row in ranking.iterrows():
    pid = row['id_produto']
    clientes_prod = set(df_vendas[df_vendas['id_product'] == pid]['id_client'])
    co_compra = clientes_gps & clientes_prod
    print(f"  [{pid}] {row['produto_nome'][:45]:<45} "
          f"co-compra: {len(co_compra)}/{len(clientes_gps)} clientes "
          f"| sim: {row['similaridade']}")