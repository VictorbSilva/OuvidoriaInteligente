"""Ouvidoria Inteligente — Entrega 4: buscador semântico em Streamlit.

Abas:
  🔍 Busca Semântica: descrição livre → as top-k manifestações mais similares, com score e cor
                      (🟢 > 0,7 · 🟡 > 0,5 · 🔴 demais);
  📋 Base Completa:   tabela com as 40 manifestações e botão que gera a matriz de similaridade;
  🌐 Espaço Vetorial: projeção 2D (PCA ou t-SNE) colorida pela categoria oficial, com métricas e
                      comentário sobre se os clusters semânticos coincidem com as categorias;
  🧩 Chunking:        cola-se uma manifestação longa, escolhe-se estratégia e parâmetros, e o app
                      mostra os chunks e os embeddings deles.
Sidebar: modelo de embedding e top-k (padrão 5).

Rodar:  streamlit run app_ouvidoria.py
O modelo é baixado do Hugging Face na primeira execução. Para abrir a interface sem rede, defina
OUVIDORIA_EMBEDDER=falso (usa um embedder determinístico local; os scores deixam de ser semânticos).
"""

import numpy as np
import pandas as pd

from ouvidoria import (
    CATEGORIAS,
    MODELO_PADRAO,
    MODELOS,
    carregar_manifestacoes,
    carregar_modelo,
    cor_do_score,
    dividir_em_chunks,
    gerar_embeddings,
    matriz_similaridade,
)

TOP_K_PADRAO = 5
ABAS = ["🔍 Busca Semântica", "📋 Base Completa", "🌐 Espaço Vetorial", "🧩 Chunking"]
ESTRATEGIAS = {
    "Recursiva (RecursiveCharacterTextSplitter)": "recursiva",
    "Tamanho fixo (CharacterTextSplitter)": "fixa",
}
CORES_CATEGORIA = {
    "infraestrutura": "#1f77b4", "saúde": "#d62728", "segurança": "#7f7f7f", "educação": "#ff7f0e", "meio ambiente": "#2ca02c",
}
CONSULTA_EXEMPLO = "rua cheia de buracos perto da avenida"


# ----------------------------------------------------------------------------- lógica (independente do Streamlit)
def ranking(consulta_emb: np.ndarray, embeddings: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Índices das top_k linhas mais similares à consulta (cosseno, embeddings normalizados), em ordem decrescente."""
    sims = embeddings @ consulta_emb
    ordem = np.argsort(-sims, kind="stable")[:top_k]
    return [(int(i), float(sims[i])) for i in ordem]


def tabela_resultados(df: pd.DataFrame, resultados: list[tuple[int, float]]) -> pd.DataFrame:
    """Resultados da busca com posição, cor, score e os campos da manifestação."""
    return pd.DataFrame([
        {"Posição": pos, "Cor": cor_do_score(s), "Score": round(s, 3), "ID": df.at[i, "id"],
         "Categoria": df.at[i, "categoria_oficial"], "Data": df.at[i, "data"], "Texto": df.at[i, "texto"]}
        for pos, (i, s) in enumerate(resultados, 1)
    ])


def pares_mais_similares(S: np.ndarray, ids: list[str], n: int = 10) -> pd.DataFrame:
    """Os n pares distintos (i < j) de maior similaridade."""
    i, j = np.triu_indices(len(ids), 1)
    ordem = np.argsort(-S[i, j], kind="stable")[:n]
    return pd.DataFrame([{"Par": f"{ids[i[k]]} × {ids[j[k]]}", "Similaridade": round(float(S[i[k], j[k]]), 3)} for k in ordem])


def projetar_2d(embeddings: np.ndarray, metodo: str) -> np.ndarray:
    """Coordenadas 2D por PCA ou t-SNE (t-SNE com perplexidade ajustada ao nº de pontos e semente fixa)."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    if len(embeddings) < 3:
        return np.column_stack([np.arange(len(embeddings), dtype=float), np.zeros(len(embeddings))])
    if metodo == "PCA":
        return PCA(n_components=2, random_state=0).fit_transform(embeddings)
    if metodo == "t-SNE":
        perplexidade = min(10, len(embeddings) - 1)
        return TSNE(n_components=2, perplexity=perplexidade, init="pca", random_state=42).fit_transform(embeddings)
    raise ValueError(f"método desconhecido: {metodo}")


def comparar_com_categorias(embeddings: np.ndarray, categorias: list[str]) -> dict:
    """Mede se os grupos semânticos coincidem com as categorias oficiais.

    - vizinho: fração das manifestações cujo vizinho mais próximo (cosseno) tem a mesma categoria;
    - silhueta: silhueta das categorias no espaço original (cosseno): perto de 1 = grupos separados, perto de 0 = misturados;
    - ari e pureza: K-Means com k = nº de categorias comparado às categorias (ARI: 1 = idêntico, 0 = acaso);
    - contingencia: cluster do K-Means × categoria.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    cat = np.asarray(categorias)
    S = embeddings @ embeddings.T
    np.fill_diagonal(S, -np.inf)
    vizinho = float((cat[np.argmax(S, axis=1)] == cat).mean())
    k = len(set(categorias))
    clusters = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(embeddings)
    contingencia = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(cat, name="categoria"))
    return {
        "vizinho": vizinho,
        "silhueta": float(silhouette_score(embeddings, cat, metric="cosine")),
        "ari": float(adjusted_rand_score(cat, clusters)),
        "pureza": float(contingencia.max(axis=1).sum() / len(cat)),
        "contingencia": contingencia,
        "clusters": clusters,
    }


def chunks_com_embeddings(texto: str, estrategia: str, chunk_size: int, chunk_overlap: int, modelo) -> tuple[list[str], np.ndarray]:
    """Divide o texto e devolve os chunks e a matriz de embeddings (uma linha por chunk)."""
    chunks = dividir_em_chunks(texto, chunk_size, chunk_overlap, estrategia)
    return chunks, gerar_embeddings(chunks, modelo) if chunks else np.zeros((0, 0))


# ----------------------------------------------------------------------------- interface
def main():
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")

    @st.cache_resource(show_spinner="Carregando o modelo de embeddings…")
    def modelo_cache(nome: str):
        return carregar_modelo(nome)

    @st.cache_data(show_spinner=False)
    def manifestacoes_cache() -> pd.DataFrame:
        return pd.DataFrame(carregar_manifestacoes())

    @st.cache_data(show_spinner="Gerando embeddings das manifestações…")
    def embeddings_cache(nome: str, textos: tuple[str, ...]) -> np.ndarray:
        return gerar_embeddings(list(textos), modelo_cache(nome))

    @st.cache_data(show_spinner="Projetando em 2D…")
    def projecao_cache(nome: str, metodo: str, textos: tuple[str, ...]) -> np.ndarray:
        return projetar_2d(embeddings_cache(nome, textos), metodo)

    @st.cache_data(show_spinner="Comparando grupos e categorias…")
    def categorias_cache(nome: str, textos: tuple[str, ...], categorias: tuple[str, ...]) -> dict:
        return comparar_com_categorias(embeddings_cache(nome, textos), list(categorias))

    df = manifestacoes_cache()
    textos = tuple(df["texto"])

    st.title("🏛️ Ouvidoria Inteligente")
    st.caption("Triagem semântica de manifestações cidadãs: busca por significado, visão da base, "
               "espaço vetorial e chunking de relatos longos.")

    # ---------------------------------------------------------------- sidebar
    st.sidebar.header("⚙️ Configurações")
    nome_modelo = st.sidebar.selectbox("Modelo de embedding", MODELOS, index=MODELOS.index(MODELO_PADRAO), key="modelo",
                                       format_func=lambda m: m.split("/")[-1])
    top_k = st.sidebar.slider("Top-k (resultados da busca)", 1, 10, TOP_K_PADRAO, key="top_k")
    modelo = modelo_cache(nome_modelo)
    emb = embeddings_cache(nome_modelo, textos)
    st.sidebar.markdown(f"**{len(df)}** manifestações · embeddings de **{emb.shape[1]}** dimensões")
    st.sidebar.markdown("**Legenda dos scores**\n\n🟢 acima de 0,7 · 🟡 acima de 0,5 · 🔴 demais")
    st.sidebar.caption("O modelo multilíngue (padrão) foi treinado com português; o all-MiniLM-L6-v2 é "
                       "só inglês e serve de comparação.")

    aba_busca, aba_base, aba_espaco, aba_chunk = st.tabs(ABAS)

    # ---------------------------------------------------------------- 🔍 busca
    with aba_busca:
        st.subheader("Descreva o problema com suas palavras")
        consulta = st.text_area("Descrição livre", value=CONSULTA_EXEMPLO, height=90, key="consulta",
                                placeholder="Ex.: falta médico no posto do meu bairro")
        if not consulta.strip():
            st.info("Digite uma descrição para buscar manifestações parecidas.")
        else:
            q = gerar_embeddings([consulta], modelo)[0]
            resultados = tabela_resultados(df, ranking(q, emb, top_k))
            st.markdown(f"**As {len(resultados)} manifestações mais similares**")
            for r in resultados.itertuples():
                with st.container(border=True):
                    st.markdown(f"{r.Cor} **#{r.Posição} · {r.ID}** · score **{r.Score:.3f}** · "
                                f"{r.Categoria} · {r.Data}")
                    st.write(r.Texto)
            st.caption("Score = similaridade de cosseno entre os embeddings da descrição e da manifestação.")

    # ---------------------------------------------------------------- 📋 base
    with aba_base:
        st.subheader("Base completa")
        filtro = st.multiselect("Filtrar por categoria", CATEGORIAS, default=CATEGORIAS, key="filtro_categoria")
        tabela = df[df["categoria_oficial"].isin(filtro)].assign(caracteres=lambda d: d["texto"].str.len())
        st.dataframe(tabela.rename(columns={"categoria_oficial": "categoria"}), hide_index=True, width="stretch")
        st.caption(f"{len(tabela)} de {len(df)} manifestações exibidas.")

        if st.button("🧮 Gerar matriz de similaridade", key="gerar_matriz"):
            st.session_state["matriz_modelo"] = nome_modelo
        if st.session_state.get("matriz_modelo") == nome_modelo:
            S = matriz_similaridade(emb)
            ids = df["id"].tolist()
            fig = go.Figure(go.Heatmap(z=S, x=ids, y=ids, colorscale="Blues", zmin=0, zmax=1,
                                       hovertemplate="%{y} × %{x}<br>cosseno = %{z:.3f}<extra></extra>"))
            fig.update_layout(height=750, title=f"Similaridade de cosseno entre as {len(ids)} manifestações",
                              yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, width="stretch")
            st.markdown("**Pares mais similares** (candidatos a duplicata; na Entrega 2 o limiar escolhido foi 0,70)")
            st.dataframe(pares_mais_similares(S, ids), hide_index=True)

    # ---------------------------------------------------------------- 🌐 espaço vetorial
    with aba_espaco:
        st.subheader("Espaço vetorial das manifestações")
        metodo = st.radio("Redução de dimensionalidade", ["PCA", "t-SNE"], horizontal=True, key="reducao")
        P = projecao_cache(nome_modelo, metodo, textos)
        comp = categorias_cache(nome_modelo, textos, tuple(df["categoria_oficial"]))
        pontos = df.assign(x=P[:, 0], y=P[:, 1], cluster=[f"cluster {c}" for c in comp["clusters"]])
        fig = px.scatter(pontos, x="x", y="y", color="categoria_oficial", text="id", color_discrete_map=CORES_CATEGORIA,
                         hover_data={"texto": True, "cluster": True, "x": False, "y": False},
                         labels={"categoria_oficial": "categoria oficial"})
        fig.update_traces(textposition="top center", marker={"size": 11})
        fig.update_layout(height=560, title=f"{metodo} dos embeddings ({nome_modelo.split('/')[-1]})")
        st.plotly_chart(fig, width="stretch")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vizinho mais próximo da mesma categoria", f"{comp['vizinho']:.0%}")
        c2.metric("Silhueta das categorias", f"{comp['silhueta']:.3f}")
        c3.metric("K-Means × categorias (ARI)", f"{comp['ari']:.2f}")
        c4.metric("Pureza dos clusters", f"{comp['pureza']:.0%}")
        with st.expander("Clusters do K-Means (k = 5) × categorias oficiais"):
            st.dataframe(comp["contingencia"])

        st.markdown(
            "#### Os clusters semânticos coincidem com as categorias?\n"
            "**Em parte.** Com o modelo multilíngue (padrão), 70% das manifestações têm como vizinho mais próximo "
            "uma da mesma categoria, e o K-Means com 5 grupos reproduz as categorias só parcialmente (ARI ≈ 0,49, "
            "pureza ≈ 75%).\n\n"
            "- **Coincidem:** *saúde* forma um grupo próprio, com todas as 8 manifestações e mais nenhuma, e "
            "*educação* e *segurança* também têm núcleos puros. O vocabulário dessas categorias é exclusivo (posto, médico, "
            "consulta; escola, alunos; assalto, drogas, ronda).\n"
            "- **Não coincidem:** *infraestrutura* e *meio ambiente* se espalham por dois grupos mistos. Um reúne "
            "problemas de via pública: buraco, semáforo e lâmpada, mas também galho sobre a fiação e som automotivo "
            "(*meio ambiente*) e carros em alta velocidade (*segurança*). O outro junta lixo, esgoto e goteira com "
            "4 dos 5 relatos longos, que são de quatro categorias diferentes.\n"
            "- **Por quê:** a categoria oficial é administrativa (qual secretaria resolve), enquanto o embedding "
            "agrupa pelo cenário descrito (rua, trânsito, água e lixo). Os relatos longos se atraem entre si "
            "pelo tom de denúncia formal (“já abrimos protocolo”, “pedimos providências”), independentemente do tema.\n"
            "- **Comparação de modelos:** com o all-MiniLM-L6-v2 (só inglês) a concordância cai (57% no vizinho "
            "mais próximo, ARI ≈ 0,31). Troque o modelo na barra lateral para ver as métricas mudarem.\n\n"
            "Os agrupamentos semânticos servem para sugerir a categoria e achar temas relacionados, mas não substituem "
            "a classificação oficial nas fronteiras entre secretarias."
        )
        st.caption("PCA e t-SNE mostram uma sombra 2D de 384 dimensões; as métricas acima são calculadas no espaço original.")

    # ---------------------------------------------------------------- 🧩 chunking
    with aba_chunk:
        st.subheader("Chunking de manifestações longas")
        longas = df.assign(n=df["texto"].str.len()).nlargest(5, "n")
        opcoes = {f"{r.id} · {r.categoria_oficial} · {r.n} caracteres": r.texto for r in longas.itertuples()}
        exemplo = st.selectbox("Carregar uma das 5 manifestações mais longas (ou cole outra abaixo)", list(opcoes),
                               key="exemplo_longo")
        texto = st.text_area("Manifestação longa", value=opcoes[exemplo], height=180, key=f"texto_longo_{exemplo}")
        c1, c2, c3 = st.columns(3)
        estrategia_rotulo = c1.selectbox("Estratégia", list(ESTRATEGIAS), key="estrategia")
        chunk_size = c2.slider("chunk_size (caracteres)", 50, 800, 300, 10, key="chunk_size")
        chunk_overlap = c3.slider("chunk_overlap (caracteres)", 0, 300, 100, 10, key="chunk_overlap")
        st.caption("Padrão 300/100: a configuração que melhor preservou o sentido das denúncias na Entrega 3.")

        if not texto.strip():
            st.info("Cole uma manifestação para dividir em chunks.")
        elif chunk_overlap >= chunk_size:
            st.error("O overlap precisa ser menor que o chunk_size.")
        else:
            chunks, E = chunks_com_embeddings(texto, ESTRATEGIAS[estrategia_rotulo], chunk_size, chunk_overlap, modelo)
            st.success(f"{len(chunks)} chunk(s) · embeddings {E.shape[0]} × {E.shape[1]}")
            for k, c in enumerate(chunks, 1):
                with st.expander(f"Chunk {k} · {len(c)} caracteres", expanded=len(chunks) <= 6):
                    st.write(c)

            st.markdown("**Embeddings dos chunks** (primeiras 12 dimensões de cada vetor)")
            st.dataframe(pd.DataFrame(E[:, :12], index=[f"C{k}" for k in range(1, len(chunks) + 1)],
                                      columns=[f"d{d}" for d in range(min(12, E.shape[1]))]).round(3))

            col_a, col_b = st.columns(2)
            rotulos = [f"C{k}" for k in range(1, len(chunks) + 1)]
            Sc = matriz_similaridade(E)
            fig = go.Figure(go.Heatmap(z=Sc, x=rotulos, y=rotulos, colorscale="Blues", zmin=0, zmax=1,
                                       text=np.round(Sc, 2), texttemplate="%{text}"))
            fig.update_layout(title="Similaridade entre os chunks", yaxis={"autorange": "reversed"}, height=420)
            col_a.plotly_chart(fig, width="stretch")

            # os chunks projetados junto com as 40 manifestações: mostra de qual tema cada chunk ficou perto
            P = projetar_2d(np.vstack([emb, E]), "PCA")
            base = df.assign(x=P[: len(df), 0], y=P[: len(df), 1], tipo="manifestação")
            pedacos = pd.DataFrame({"x": P[len(df):, 0], "y": P[len(df):, 1], "id": rotulos, "texto": chunks})
            fig = px.scatter(base, x="x", y="y", color="categoria_oficial", color_discrete_map=CORES_CATEGORIA,
                             hover_data={"id": True, "x": False, "y": False}, opacity=0.35,
                             labels={"categoria_oficial": "categoria oficial"})
            fig.add_trace(go.Scatter(x=pedacos["x"], y=pedacos["y"], mode="markers+text+lines", text=pedacos["id"],
                                     textposition="top center", name="chunks", marker={"size": 13, "color": "black"},
                                     line={"dash": "dot", "color": "black"}, hovertext=pedacos["texto"]))
            fig.update_layout(title="Chunks no espaço das manifestações (PCA)", height=420)
            col_b.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()
