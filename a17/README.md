# a17 — Ouvidoria Inteligente · Entrega 4: Buscador Semântico + App Streamlit

**Fonte:** `aulas/Desafio_Ouvidoria_Inteligente.docx`, seção 4, Entrega 4 (3,0 pts).

## Enunciado (resumo)

> Desenvolva uma aplicação Streamlit (`app_ouvidoria.py`) com as abas 🔍 Busca Semântica (descrição livre → as 5 manifestações mais similares, com score e cor: 🟢 > 0,7, 🟡 > 0,5, 🔴 demais), 📋 Base Completa (tabela e botão para gerar a matriz de similaridade), 🌐 Espaço Vetorial (2D com PCA/t-SNE selecionável, colorido pela categoria oficial, com comentário sobre se os clusters coincidem com as categorias) e 🧩 Chunking (colar uma manifestação longa, escolher estratégia e parâmetros e ver os chunks e seus embeddings). Executável com `streamlit run app_ouvidoria.py`, com sidebar para escolher o modelo de embedding e o top-k.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `app_ouvidoria.py` | O app. As funções de lógica (`ranking`, `tabela_resultados`, `projetar_2d`, `comparar_com_categorias`, `chunks_com_embeddings`) ficam no topo e a interface em `main()`. |
| `ouvidoria.py` | Módulo compartilhado das entregas: carga dos dados, embeddings, cores dos scores e splitters do LangChain. |
| `manifestacoes.json` | As 40 manifestações do desafio. |

## O app

| Onde | O que tem |
|---|---|
| **Sidebar** | Modelo de embedding (`paraphrase-multilingual-MiniLM-L12-v2`, padrão, ou `all-MiniLM-L6-v2`) e top-k de 1 a 10 (padrão 5), mais a legenda das cores. |
| **🔍 Busca Semântica** | Descrição livre → as top-k manifestações por similaridade de cosseno, cada uma com posição, cor, score, ID, categoria, data e texto. Exemplo: “rua cheia de buracos perto da avenida” traz M003 (🟢 0,737) e M017 (0,630), que não usam a palavra “buracos”. |
| **📋 Base Completa** | Tabela das 40 manifestações com filtro por categoria. O botão **🧮 Gerar matriz de similaridade** mostra o heatmap 40 × 40 interativo e os 10 pares mais similares. |
| **🌐 Espaço Vetorial** | PCA ou t-SNE (rádio), pontos coloridos pela categoria oficial com o texto no hover. Mostra quatro métricas calculadas no espaço original (vizinho mais próximo da mesma categoria, silhueta, ARI e pureza do K-Means com k = 5), a tabela cluster × categoria e o comentário. |
| **🧩 Chunking** | Carrega uma das 5 manifestações mais longas ou aceita texto colado. Escolhe a estratégia (recursiva ou tamanho fixo), o `chunk_size` e o `chunk_overlap` (padrão 300/100, o melhor da Entrega 3). Mostra os chunks, as 12 primeiras dimensões dos embeddings, a similaridade entre os chunks e os chunks projetados junto com as 40 manifestações. |

### Os clusters coincidem com as categorias? (comentário da aba 🌐)

**Em parte.** Com o modelo multilíngue, 70% das manifestações têm como vizinho mais próximo uma da mesma categoria, e o K-Means reproduz as categorias só parcialmente (ARI ≈ 0,49, pureza 75%). *Saúde* forma um grupo puro e completo (8 de 8), e *educação* e *segurança* têm núcleos puros. Já *infraestrutura* e *meio ambiente* se espalham por dois grupos mistos: o de via pública (buraco, semáforo, lâmpada, galho sobre a fiação, som automotivo) e o de água e lixo, que também reúne 4 dos 5 relatos longos. A categoria oficial é administrativa, e o embedding agrupa pelo cenário descrito. Com o `all-MiniLM-L6-v2` (só inglês), a concordância cai para 57% e ARI ≈ 0,31.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run app_ouvidoria.py
```

Na primeira execução, os modelos são baixados do Hugging Face (≈ 470 MB o multilíngue, ≈ 90 MB o inglês). Para abrir a interface sem rede, rode com `OUVIDORIA_EMBEDDER=falso` (PowerShell: `$env:OUVIDORIA_EMBEDDER="falso"`). O app passa a usar um embedder determinístico local, então os scores deixam de ter sentido semântico.

## Suposições

- **Top-k:** o enunciado fixa “as 5 manifestações mais similares” e pede top-k configurável. O padrão é 5, ajustável de 1 a 10.
- **Cores:** “> 0,7” e “> 0,5” são estritos: um score de exatamente 0,7 é 🟡, e um de 0,5 é 🔴.
- **Modelos:** a seção 7 sugere comparar pelo menos dois modelos. O app oferece o multilíngue (padrão) e o `all-MiniLM-L6-v2`, citado nos materiais de apoio, como comparação.
- **Comentário da aba 🌐:** o texto vale para o modelo padrão. As métricas acima dele são recalculadas para o modelo escolhido.
- **Cache:** `st.cache_resource` para o modelo e `st.cache_data` para os embeddings, as projeções e as métricas, como recomenda a seção 7.
- **Matriz de similaridade:** fica visível depois do clique e é recalculada se o modelo mudar (o clique vale para o modelo em uso).
