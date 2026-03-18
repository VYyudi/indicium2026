# README - Projeto LH-NAUTICALS

Este projeto gera análise de vendas e um dashboard interativo com dados tratados. O código principal está em `pipeline.py`.

## Pré-requisitos
1. Python 3.10+ instalado
2. Bibliotecas: `pandas`, `scikit-learn`
3. Arquivos de dados em `datasets/` (já incluídos)


```powershell
python -m pip install pandas scikit-learn
```

## Executar pipeline
Rode uma vez para gerar `data.json` usado pelo dashboard:

```powershell
python pipeline.py
```

Saída esperada:
- `data.json` gerado
- Impressões no terminal com volume de produtos, vendas, lucro e previsão

## Visualizar dashboard
1. Abra `dashboard.html` no VS Code
2. Instale e ative Live Server
3. Clique com o botão direito em `dashboard.html` → `Open with Live Server`

## Arquivos importantes
- `pipeline.py`: tratamento, modelagem e exportação JSON.
- `dashboard.html`: visualização front-end.
- `tratamento_produtos.py`: script de limpeza de produtos.
- `prejuizo.py`: consulta de câmbio e cálculo de prejuízo.

## Observação
Se a API de câmbio do Banco Central falhar (400), o arquivo `prejuizo.py` precisa de ajuste no formato de datas da consulta (ex.: `2023-01-01`).
