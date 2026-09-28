# a16 — Ouvidoria Inteligente · Entrega 3: Chunking de Manifestações Longas

**Fonte:** `aulas/Desafio_Ouvidoria_Inteligente.docx`, seção 4, Entrega 3 (2,0 pts).

## Enunciado (resumo)

> As 5 manifestações mais longas devem ser divididas em chunks usando a estratégia `RecursiveCharacterTextSplitter` do LangChain. Teste pelo menos duas configurações diferentes de (chunk_size, chunk_overlap) e responda: como o overlap influencia a coesão semântica entre chunks consecutivos? Qual configuração preserva melhor o sentido das denúncias? Justifique com exemplos. Visualize os chunks no espaço 2D (PCA ou t-SNE). Os chunks de uma mesma manifestação ficam próximos?

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `chunking_manifestacoes.ipynb` | O notebook, já executado: seleção das 5 mais longas, 4 configurações de chunking, métricas, exemplos, projeções PCA e t-SNE e as respostas às três perguntas. |
| `ouvidoria.py` | Módulo compartilhado das entregas. Aqui é usado o `dividir_em_chunks` (splitter do LangChain) e a geração de embeddings. |
| `manifestacoes.json` | As 40 manifestações do desafio. |

## Configurações testadas

Dois tamanhos, cada um com e sem overlap, para isolar o efeito do overlap:

| Config. | (chunk_size, chunk_overlap) | Chunks | Coesão consecutiva | Chunks que apontam para a própria manifestação |
|---|---|---|---|---|
| A | (150, 0) | 34 | 0,223 | 74% |
| B | (150, 50) | 34 | 0,288 | 74% |
| C | (300, 0) | 15 | 0,480 | 87% |
| **D** | **(300, 100)** | **17** | **0,597** | **88%** |

## Respostas (resumo)

- **Overlap e coesão:** o overlap aumenta a coesão entre chunks consecutivos (A→B e C→D). No `RecursiveCharacterTextSplitter`, porém, ele repete pedaços inteiros (frases, ou palavras quando a frase foi quebrada), não uma janela de caracteres. Um overlap menor que as frases do texto não muda nada: com `chunk_size` 250, overlap 0 e 60 dão os mesmos chunks.
- **Melhor configuração:** D (300, 100). Cada chunk tem de 2 a 3 frases completas, não há fragmentos como “estavam no trabalho.” ou “irregular.”, e é a configuração com mais chunks que, sozinhos, ainda apontam para a denúncia de origem.
- **Proximidade em 2D:** com D, sim: 82% dos chunks têm como vizinho mais próximo outro chunk da mesma manifestação, e no t-SNE cada denúncia forma uma trilha separada. Com chunks de uma frase (A, B), não: os chunks se agrupam pelo assunto da frase (burocracia, danos à saúde), não pela denúncia.

## Como rodar

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace chunking_manifestacoes.ipynb   # ou abra com jupyter notebook
```

Na primeira execução, o modelo `paraphrase-multilingual-MiniLM-L12-v2` (≈ 470 MB) é baixado do Hugging Face.

## Suposições

- **"5 mais longas":** medidas em caracteres (o esquema do enunciado limita o texto em caracteres). São M025, M020, M013, M005 e M032, com 589 a 682 caracteres; a 6ª tem 173.
- **Separadores:** parágrafo, linha, frase (`". "`), palavra e caractere, com o ponto final mantido no fim da frase (`keep_separator="end"`). As manifestações não têm quebras de linha, então o corte útil começa na frase.
- **"Preservar o sentido":** medido de três formas, além dos exemplos: fragmentos (chunks que começam ou terminam no meio da frase), similaridade do chunk com a manifestação inteira, e se o chunk, sozinho, tem como manifestação mais similar entre as 40 a sua própria origem.
- **Projeção:** o enunciado pede PCA ou t-SNE; o notebook mostra as duas. O t-SNE usa perplexidade 5 e `random_state=42`, para ser reproduzível.
