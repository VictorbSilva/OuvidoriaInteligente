# Ouvidoria Inteligente — Triagem Semântica de Manifestações Cidadãs

Desafio prático de NLP aplicado da disciplina **Tendências em Ciência da Computação** (Prof. Me. Ricardo Roberto de Lima). Tema: representações vetoriais, busca semântica e chunking.

A ouvidoria de um município recebe milhares de manifestações por mês e hoje faz a triagem por palavras-chave. Por isso não detecta duplicatas ("asfalto esburacado" × "rua com buraco"), não agrupa temas relacionados e perde contexto em textos longos. Este projeto resolve esses problemas com representações vetoriais modernas.

| Pasta | Entrega | Arquivo principal |
|---|---|---|
| [`a14/`](a14/) | 1 — Análise comparativa de representações (BoW, TF-IDF, embeddings) | `análise_comparativa.ipynb` |
| [`a15/`](a15/) | 2 — Detecção de duplicatas (função, heatmap, escolha do limiar, falsos positivos e negativos) | `deteccao_duplicatas.ipynb` |

Cada pasta é independente e tem o próprio `README.md` (enunciado, como rodar e suposições), `requirements.txt` e uma cópia dos dados (`manifestacoes.json`, 40 manifestações montadas no esquema do enunciado). Cada entrega foi aprovada numa auditoria independente antes da publicação.
