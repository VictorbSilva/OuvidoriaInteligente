# a14 — Ouvidoria Inteligente · Entrega 1: Análise Comparativa de Representações

**Fonte:** `aulas/Desafio_Ouvidoria_Inteligente.docx`, seção 4, Entrega 1 (2,5 pts).

## Enunciado (resumo)

> Utilizando o corpus das 40 manifestações, gere representações BoW, TF-IDF e Embeddings (modelo à sua escolha, preferencialmente multilíngue). Para os pares M003 × M017, M008 × M022 e M008 × M031, calcule a similaridade de cosseno em cada representação e discuta os resultados. Entregue um notebook (.ipynb) com o código, as tabelas comparativas e um parágrafo conclusivo sobre as limitações de cada abordagem.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `análise_comparativa.ipynb` | O notebook, já executado: as três representações, a tabela pares × representações, os termos em comum de cada par, a discussão e a conclusão sobre as limitações. |
| `ouvidoria.py` | Funções compartilhadas pelas entregas da Ouvidoria: carga dos dados, stopwords em português e embeddings. |
| `manifestacoes.json` | As 40 manifestações, montadas para o desafio (ver Suposições). |
| `duplicatas_reais.json` | Gabarito de duplicatas do conjunto, usado na Entrega 2. |

## Resultado

| Par | BoW | TF-IDF | Embeddings |
|---|---|---|---|
| M003 × M017 (mesmo buraco) | 0,143 | 0,133 | 0,715 |
| M008 × M022 (mesmo posto sem médico) | 0,167 | 0,159 | 0,848 |
| M008 × M031 (temas diferentes) | 0,000 | 0,000 | 0,225 |

Nas representações esparsas, os pares duplicados só compartilham nomes próprios (“sete”/“setembro”, “são”/“josé”), e por isso ficam quase tão distantes quanto o par sem relação. Os embeddings separam bem os casos.

## Como rodar

```bash
pip install -r requirements.txt
jupyter notebook análise_comparativa.ipynb
# ou, sem abrir o navegador:
jupyter nbconvert --to notebook --execute --inplace análise_comparativa.ipynb
```

Na primeira execução, o modelo `paraphrase-multilingual-MiniLM-L12-v2` (≈ 470 MB) é baixado do Hugging Face.

## Suposições

- **Dados:** o `manifestacoes.json` é produzido pelo aluno. Este conjunto segue o esquema da seção 2:
  - 40 manifestações, M001 a M040, com data AAAA-MM-DD e texto de 116 a 682 caracteres;
  - 5 categorias oficiais;
  - 6 duplicatas semânticas (15%);
  - 5 textos com mais de 500 caracteres;
  - os trechos citados nos pares ("buraco enorme na Av. Brasil", "asfalto todo esburacado da avenida principal", "posto de saúde sem médico", "falta atendimento no PSF", "lâmpada queimada na praça"), dentro de relatos completos. Frases curtas assim, sozinhas, ficariam abaixo do mínimo de 50 caracteres do esquema.

  Outro conjunto no mesmo esquema funciona trocando o arquivo e reexecutando o notebook.
- **Modelo:** `paraphrase-multilingual-MiniLM-L12-v2`, o primeiro sugerido. O segundo, `BAAI/bge-small-pt-v1.5`, não existe no Hugging Face.
- **Stopwords:** BoW e TF-IDF usam uma lista de stopwords em português, porque o scikit-learn só traz a lista em inglês.
- **Normalização:** os embeddings são normalizados (norma 1), então o produto escalar é o cosseno.
