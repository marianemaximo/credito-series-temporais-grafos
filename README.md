# Projeto 02 — Inadimplência de crédito: séries temporais e grafos de quantis

**Pergunta:** a topologia do *grafo de quantis* (QG) de Campanharo et al. (2011) contém informação útil para **prever a inadimplência**, além do que já capturam a sazonalidade e a inércia da própria série?

**Resposta curta (honesta):** o QG **descreve bem** a dinâmica da série, mas suas features **não melhoram a previsão**. O único ganho robusto vem da **sazonalidade**, e só em 1 mês à frente.

## Dados
Banco Central do Brasil, SGS **21082** — *Inadimplência da carteira de crédito – Total* (% da carteira do SFN com atraso > 90 dias). Mensal, **mar/2011 a set/2026 (187 observações)**. É um indicador **agregado do sistema**, não o risco de um tomador individual. Arquivo em `data/raw/bcdata.sgs.21082.csv`.

## O método
Campanharo, Sirer, Malmgren, Ramos & Amaral (2011) propõem mapear uma série em uma rede: a série é dividida em *Q* quantis, cada quantil é um nó e o peso do arco *i → j* é a probabilidade de um ponto no quantil *i* ser seguido por um no quantil *j*. Medidas da rede (persistência/self-loop, comprimento médio de salto, entropia, modularidade, caminho médio) caracterizam a dinâmica da série.

> As funções de métricas de rede e a previsão markoviana em `src/qgraph.py` são **implementações próprias** sobre essa construção. A estimativa de Hurst segue a ideia descrita no resumo de Campanharo & Ramos (2016) e foi validada em fBm simulado; **não é uma reprodução linha a linha do artigo**.

## O que foi feito
| Etapa | Detalhe |
|---|---|
| EDA | ADF, autocorrelação, sazonalidade (Kruskal-Wallis) |
| QG | grafo e matriz *P*; **surrogates** (embaralhado e fase, 1.000 cada); **Hurst via QG** com calibração por simulação; métricas em janelas deslizantes |
| Previsão | walk-forward (91 previsões, fev/2019–set/2026), horizontes de **1, 3 e 6 meses**, sem vazamento (há teste automático) |
| Baselines | passeio aleatório, *drift*, média por mês, AR+sazonal, ARIMA(1,1,1) |
| Modelos com QG | Ridge e Random Forest com 10 medidas do QG; QG-Markov como previsor direto |
| Rigor | Diebold-Mariano (HLN), bootstrap em blocos, **correção de Holm**, robustez em 30 configurações (janela × Q × horizonte) |

## Resultados principais
- **Sazonalidade:** a variação média depende do mês (Kruskal-Wallis p ≈ 4·10⁻¹⁴): quedas em junho (−0,14 p.p.) e dezembro (−0,10), altas em janeiro (+0,11) e abril (+0,09).
- **QG vs. ruído:** o QG da série em nível é muito mais persistente que surrogates embaralhados e de fase (p ≈ 0,001); o das **variações mensais é indistinguível de ruído**.
- **Hurst via QG:** 0,69 observado, ≈ 0,81 após calibração (compatível com 0,65–0,95): série persistente.
- **Previsão (RMSE relativo ao passeio aleatório, h = 1):** média por mês **0,82**; Random Forest 0,80; AR+sazonal+QG (Ridge) 0,99; QG-Markov 2,68.
- **O QG ajuda?** Em **nenhuma** das 6 comparações pré-especificadas. Razão mediana de RMSE (com QG ÷ sem QG) = **1,16** em 30 configurações. Piora do Ridge em h = 3 e 6: IC de bootstrap exclui 1, mas **p ajustado por Holm = 0,21 e 0,15** (não significativo).
- **Sazonal vs. passeio aleatório (h = 1):** razão 0,82 (IC 95% 0,74–0,92), p bruto 0,013, **p Holm = 0,079**: forte indício, não significativo a 5% com 6 testes. Acerto de direção: **77% vs. 61%** da taxa-base (p = 0,001, teste único).
- **h = 3 e 6:** nenhum modelo supera o passeio aleatório de forma significativa.
- **Referência a partir de set/2026 (4,98%):** out/2026 ≈ 5,02% (80%: 4,91–5,16); dez/2026 e mar/2027 ≈ 4,98%, com intervalos mais largos. Não é recomendação.

## Limitações
Uma única série; 91 previsões (poder limitado); escolhas de janela e Q; período de teste com regimes incomuns (2020 e 2025–26); sem variáveis macroeconômicas. Um desvio do plano (variante "nó→nó" do QG-Markov) está marcado como **pós-hoc**. Por que o QG pode não ter ajudado são hipóteses, não testadas (série curta; o QG de nível não codifica o calendário; poucas transições por janela).

## Próximos passos
(i) variáveis exógenas (Selic, desemprego, concessões, IPCA); (ii) aplicar o QG a **séries por cliente** (atraso, uso do limite, pagamentos) como features de um *score* individual; (iii) usar o QG para **classificar regimes**; (iv) séries de maior frequência.

## Estrutura
```
projeto_credito_qg/
├── notebooks/02_credito_series_temporais_grafos.ipynb   # análise completa, executada
├── notebooks/02_credito_series_temporais_grafos.html    # leitura sem Jupyter
├── src/qgraph.py      # grafo de quantis, métricas, surrogates, Hurst
├── src/forecast.py    # features, walk-forward, DM, bootstrap, Holm
├── tests/             # 7 testes (inclui ausência de vazamento temporal)
├── reports/figures/   # figuras;  reports/metrics_credito.json
└── data/raw/bcdata.sgs.21082.csv
```

## Como rodar
```bash
pip install numpy pandas scipy scikit-learn statsmodels networkx matplotlib seaborn jupyter nbclient
python -m pytest -q tests
jupyter nbconvert --to notebook --execute --inplace notebooks/02_credito_series_temporais_grafos.ipynb   # ~4 min
```

## Referências
- Campanharo, A. S. L. O.; Sirer, M. I.; Malmgren, R. D.; Ramos, F. M.; Amaral, L. A. N. (2011). *Duality between Time Series and Networks.* PLoS ONE 6(8): e23378. doi:10.1371/journal.pone.0023378
- Campanharo, A. S. L. O.; Ramos, F. M. (2016). *Hurst exponent estimation of self-affine time series using quantile graphs.* Physica A 444: 43–48.
- Campanharo, A. S. L. O.; Doescher, E.; Ramos, F. M. (2018). *Application of Quantile Graphs to the Automated Analysis of EEG Signals.* Neural Processing Letters.
- Dados: Banco Central do Brasil, SGS 21082.
