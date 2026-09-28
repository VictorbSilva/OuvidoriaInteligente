# a18 — Ouvidoria Inteligente · Relatório Final

**Fonte:** `aulas/Desafio_Ouvidoria_Inteligente.docx`, seção 6 (Arquivos a Entregar).

## Enunciado (resumo)

> `RELATORIO.pdf` — documento de até 5 páginas sintetizando decisões, dificuldades e aprendizados.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `RELATORIO.pdf` | O relatório (3 páginas): visão geral, dados, decisões e resultados de cada uma das 4 entregas, dificuldades e aprendizados, com 2 figuras. |
| `gerar_relatorio.ipynb` | O notebook que gera o relatório. Recalcula os números das 4 entregas, desenha as figuras, escreve `RELATORIO.md` e converte em PDF. |
| `RELATORIO.md` | O texto do relatório em Markdown (gerado pelo notebook). |
| `figuras/` | As figuras do relatório: distribuição das similaridades com os limiares (Entrega 2) e t-SNE dos chunks (Entrega 3). |
| `relatorio_pdf.py` | Conversor Markdown → PDF (fpdf2). |
| `ouvidoria.py`, `manifestacoes.json`, `duplicatas_reais.json` | O módulo e os dados das entregas, para que os números sejam recalculados aqui. |

## Conteúdo do relatório

1. **Visão geral:** o problema e a decisão principal e o resultado de cada entrega.
2. **Dados:** como o `manifestacoes.json` foi montado e por que o gabarito de duplicatas fica à parte.
3. **Entrega 1:** stopwords em português, escolha do modelo e tabela dos três pares (BoW, TF-IDF e embeddings).
4. **Entrega 2:** assinatura da função, varredura de limiares, escolha de 0,70 e figura da distribuição.
5. **Entrega 3:** as 4 configurações de chunking, as medidas, por que o overlap age em frases inteiras e a figura t-SNE.
6. **Entrega 4:** arquitetura do app, caches, abas e a comparação dos grupos semânticos com as categorias.
7. **Dificuldades** e 8. **Aprendizados**.

## Como rodar

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace gerar_relatorio.ipynb   # regrava RELATORIO.md, figuras/ e RELATORIO.pdf
```

Na primeira execução, os modelos `paraphrase-multilingual-MiniLM-L12-v2` (≈ 470 MB) e `all-MiniLM-L6-v2` (≈ 90 MB) são baixados do Hugging Face.

## Suposições

- **Formato:** o enunciado pede um PDF. O PDF é gerado por um notebook para que cada número citado venha do mesmo código das entregas, e não de cópia manual.
- **Autoria:** a modalidade é individual ou em dupla. Esta entrega é individual.
- **Fonte do PDF:** Arial, quando existe no sistema (Windows). Em outro sistema, o conversor usa Helvetica e troca os caracteres fora do latin-1 (aspas curvas, travessão, setas) por equivalentes simples.
