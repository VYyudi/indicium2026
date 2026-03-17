import pandas as pd
import json

with open('datasets/custos_importacao.json', encoding='utf-8') as f:
    data = json.load(f)

# Flatten do histórico aninhado em linhas planas
rows = []
for product in data:
    for entry in product['historic_data']:
        rows.append({
            'product_id':   product['product_id'],
            'product_name': product['product_name'],
            'category':     product['category'],
            'start_date':   entry['start_date'],
            'usd_price':    entry['usd_price'],
        })

df = pd.DataFrame(rows)

# Tipagem correta
df['product_id']   = df['product_id'].astype(int)
df['product_name'] = df['product_name'].astype(str)
df['category']     = df['category'].astype(str)
df['start_date']   = pd.to_datetime(df['start_date'], format='%d/%m/%Y').dt.date
df['usd_price']    = df['usd_price'].astype(float)

df.to_csv('custos_importacao.csv', index=False)

print(f"Linhas geradas: {len(df)}")
print(f"Produtos únicos: {df['product_id'].nunique()}")
print(df.head(10))