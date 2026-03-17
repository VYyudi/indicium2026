"""
Questão — Previsão de Demanda
==============================
Produto: Motor de Popa Yamaha Evo Dash 155HP (id_product = 54)
Modelo: Baseline — Média Móvel dos últimos 7 dias
Treino: 01/01/2023 – 31/12/2023
Teste: 01/01/2024 – 31/01/2024
Métrica: MAE (Mean Absolute Error)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 1 — CARREGAMENTO E PREPARAÇÃO
df_vendas = pd.read_csv('datasets/vendas_2023_2024.csv')

def parse_date(d):
    for fmt in ['%Y-%m-%d', '%d-%m-%Y']:
        try:
            return pd.to_datetime(d, format=fmt)
        except ValueError:
            pass
    return pd.NaT

df_vendas['date'] = df_vendas['sale_date'].apply(parse_date)

PRODUCT_ID   = 54
PRODUCT_NAME = 'Motor de Popa Yamaha Evo Dash 155HP'
WINDOW = 7   # dias para a média móvel

# Filtra apenas o produto alvo
vendas_prod = (
    df_vendas[df_vendas['id_product'] == PRODUCT_ID][['date', 'qtd']]
    .groupby('date')['qtd']
    .sum()
)

# Calendário completo: 01/01/2023 → 31/01/2024
full_index = pd.date_range('2023-01-01', '2024-01-31', freq='D')

# Dias sem venda = 0 (premissa: loja aberta todos os dias)
serie = vendas_prod.reindex(full_index, fill_value=0)
serie.index.name = 'date'
serie.name = 'qtd_real'

print(f"Produto : {PRODUCT_NAME}")
print(f"Período : {serie.index.min().date()} → {serie.index.max().date()}")
print(f"Total de dias: {len(serie)}")
print(f"Dias com venda > 0: {(serie > 0).sum()}")
print(f"Qtd total vendida 2023: {serie['2023'].sum()}")
print(f"Qtd total vendida Jan/2024: {serie['2024-01'].sum()}")

# 2 — SPLIT TREINO / TESTE
treino = serie[:'2023-12-31']   # 365 dias
teste  = serie['2024-01-01':'2024-01-31']  # 31 dias

print(f"\nTreino : {treino.index.min().date()} → {treino.index.max().date()} ({len(treino)} dias)")
print(f"Teste  : {teste.index.min().date()}  → {teste.index.max().date()}  ({len(teste)} dias)")


# 3 — MODELO BASELINE: MÉDIA MÓVEL 7 DIAS
# Para cada dia de Janeiro/2024, a previsão é a média dos 7 dias
# ANTERIORES à data prevista — sem usar nenhum dado futuro (sem leakage).

previsoes = []

for data in teste.index:
    janela = serie[serie.index < data].tail(WINDOW)
    previsao = janela.mean() if len(janela) == WINDOW else janela.mean()
    previsoes.append({
        'date':     data,
        'real':     teste[data],
        'previsto': round(previsao, 2)
    })

df_result = pd.DataFrame(previsoes).set_index('date')

print("\n=== PREVISÃO DIÁRIA — JANEIRO 2024 ===")
print(df_result.to_string())


# 4 — MÉTRICA: MAE

mae = (df_result['real'] - df_result['previsto']).abs().mean()

print(f"\n=== RESULTADO ===")
print(f"MAE (Mean Absolute Error) : {mae:.4f} unidades/dia")
print(f"Média real Jan/2024       : {df_result['real'].mean():.4f} unidades/dia")
print(f"Média prevista Jan/2024   : {df_result['previsto'].mean():.4f} unidades/dia")
print(f"Erro relativo (MAE/média) : {mae / df_result['real'].mean() * 100:.1f}%" if df_result['real'].mean() > 0 else "Média real = 0")

# 5 — GRÁFICO

fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [2, 1]})
fig.patch.set_facecolor('#0a1628')

# Gráfico principal: real vs previsto 
ax1 = axes[0]
ax1.set_facecolor('#0d1f38')

# Contexto: últimos 30 dias de treino (para dar continuidade visual)
contexto = treino.tail(30)
ax1.plot(contexto.index, contexto.values,
         color='#88aabb', linewidth=1.2,
         linestyle='--', label='Histórico (dez/2023)', alpha=0.6)

ax1.plot(df_result.index, df_result['real'],
         color='#00d4b8', linewidth=2, marker='o', markersize=5,
         label='Real (jan/2024)')

ax1.plot(df_result.index, df_result['previsto'],
         color='#f5a623', linewidth=2, marker='s', markersize=4,
         linestyle='--', label='Previsto (MM7)')

# Área de erro
ax1.fill_between(df_result.index,
                 df_result['real'], df_result['previsto'],
                 alpha=0.12, color='#e8394a', label='Erro')

ax1.axvline(pd.Timestamp('2024-01-01'), color='#e8394a',
            linestyle=':', linewidth=1.5, alpha=0.7)
ax1.text(pd.Timestamp('2024-01-01'), ax1.get_ylim()[1] if ax1.get_ylim()[1] > 0 else 1,
         ' início do teste', color='#e8394a', fontsize=9, va='top')

ax1.set_title(f'Previsão de Demanda — {PRODUCT_NAME}\nBaseline: Média Móvel {WINDOW} dias | MAE = {mae:.2f} unid/dia',
              color='#f0f4f8', fontsize=12, fontweight='bold', pad=12)
ax1.set_ylabel('Qtd Vendida (unidades)', color='#8899aa')
ax1.tick_params(colors='#8899aa')
ax1.legend(facecolor='#0d1f38', edgecolor='#003d38',
           labelcolor='#f0f4f8', fontsize=10)
for spine in ax1.spines.values():
    spine.set_visible(False)
ax1.grid(axis='y', color='#1e3050', linewidth=0.7)
ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

#Gráfico de erro diário
ax2 = axes[1]
ax2.set_facecolor('#0d1f38')

erro_diario = (df_result['real'] - df_result['previsto']).abs()
bar_colors  = ['#e8394a' if e > mae else '#7a1a24' for e in erro_diario]
ax2.bar(df_result.index, erro_diario, color=bar_colors, width=0.7)
ax2.axhline(mae, color='#f5a623', linestyle='--', linewidth=1.5,
            label=f'MAE = {mae:.2f}')

ax2.set_title('Erro Absoluto Diário', color='#f0f4f8', fontsize=11, pad=8)
ax2.set_ylabel('|Real − Previsto|', color='#8899aa')
ax2.tick_params(colors='#8899aa')
ax2.legend(facecolor='#0d1f38', edgecolor='#003d38',
           labelcolor='#f0f4f8', fontsize=10)
for spine in ax2.spines.values():
    spine.set_visible(False)
ax2.grid(axis='y', color='#1e3050', linewidth=0.7)
ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.tight_layout(pad=2)
plt.savefig('grafico_previsao_demanda.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("\nGráfico salvo: grafico_previsao_demanda.png")


# 6 — ANÁLISE OBJETIVA
print("\n" + "═"*60)
print("ANÁLISE OBJETIVA")
print("═"*60)
print(f"""
a) O baseline é adequado para esse produto?

   Produto com {(serie['2023'] > 0).sum()} dias com venda em 365 dias de treino —
   ou seja, a maior parte dos dias registra qtd = 0.
   Com MAE de {mae:.2f} unidades/dia e média real de
   {df_result['real'].mean():.2f} unid/dia em janeiro,
   {"o baseline tem desempenho razoável dado a baixa frequência de vendas." if mae < 2 else "o erro é relevante em relação ao volume médio diário."}

b) Uma limitação do método:

   A média móvel de 7 dias não consegue capturar sazonalidade
   nem picos pontuais de demanda. Se houver um surto de vendas
   em um final de semana ou uma promoção específica, a janela
   de 7 dias demora a reagir — ela sempre "persegue" o passado
   recente sem antecipar padrões futuros.
""")