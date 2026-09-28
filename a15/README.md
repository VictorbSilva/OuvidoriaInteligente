# a15 — Ouvidoria Inteligente · Entrega 2: Detecção de Duplicatas

**Fonte:** `aulas/Desafio_Ouvidoria_Inteligente.docx`, seção 4, Entrega 2 (2,5 pts).

## Enunciado (resumo)

> Construa uma função `detectar_duplicatas(textos, limiar=0.85)` que receba a lista de manifestações e retorne todos os pares com similaridade acima do limiar. Justifique a escolha do limiar e apresente: a matriz de similaridade completa (heatmap); a lista de pares duplicados encontrados; uma análise de falsos positivos e falsos negativos, comparando com as duplicatas reais do dataset.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `deteccao_duplicatas.ipynb` | O notebook, já executado: a função, o heatmap 40 × 40, os pares com 0,85, a varredura de limiares com o percentil 90 sugerido, a justificativa do limiar escolhido, os pares finais e a análise de FP/FN. |
| `ouvidoria.py` | Contém `detectar_duplicatas(textos, limiar=0.85, modelo=None)` e `avaliar_deteccao` (precisão, revocação e F1), além da carga dos dados e dos embeddings. |
| `manifestacoes.json` | As 40 manifestações do desafio. |
| `duplicatas_reais.json` | Gabarito: os 6 pares de duplicatas semânticas do conjunto. |

## Resultado

| Limiar | Duplicatas detectadas (de 6) | Falsos positivos |
|---|---|---|
| 0,85 (padrão do enunciado) | 1 | 0 |
| 0,471 (percentil 90, dica da seção 7) | 6 | 72 |
| **0,70 (escolhido)** | **6** | **0** |

A menor duplicata real tem similaridade 0,715 e a maior não duplicata tem 0,660. O limiar 0,70 fica entre as duas.

## Como rodar

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace deteccao_duplicatas.ipynb   # ou abra com jupyter notebook
```

Na primeira execução, o modelo `paraphrase-multilingual-MiniLM-L12-v2` (≈ 470 MB) é baixado do Hugging Face.

## Suposições

- **Duplicatas reais:** o esquema da seção 2 não tem campo que marque duplicatas. Por isso o gabarito fica num arquivo à parte (`duplicatas_reais.json`), montado junto com as manifestações, que são produzidas pelo aluno.
- **Assinatura:** a função mantém o padrão `limiar=0.85` pedido e ganha um parâmetro opcional `modelo`, para trocar o modelo de embeddings. A comparação é “acima do limiar” (`>`), como no enunciado.
- **Limiar escolhido:** 0,70 foi calibrado neste conjunto de 40 manifestações. A folga entre os grupos é estreita, então com outro conjunto ele deve ser recalibrado com a mesma varredura. O notebook recomenda usar a detecção para sugerir duplicatas a um atendente, e não para fundi-las automaticamente.
- **Modelo:** `paraphrase-multilingual-MiniLM-L12-v2`, o mesmo da Entrega 1.
