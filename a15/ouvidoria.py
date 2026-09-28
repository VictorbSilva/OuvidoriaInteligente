"""Funções compartilhadas pelas entregas da Ouvidoria Inteligente.

Carrega as manifestações, gera representações (BoW, TF-IDF e embeddings), calcula
similaridades, detecta duplicatas e divide textos longos em chunks.

O modelo de embeddings (sentence-transformers) é baixado da internet na primeira execução.
Para testes sem rede, defina OUVIDORIA_EMBEDDER=falso: o carregador passa a devolver um
embedder determinístico (EmbedderFalso) com a mesma interface de `encode`.
"""

import hashlib
import json
import os
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

PASTA = Path(__file__).resolve().parent
MODELO_PADRAO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODELOS = [MODELO_PADRAO, "sentence-transformers/all-MiniLM-L6-v2"]
CATEGORIAS = ["infraestrutura", "saúde", "segurança", "educação", "meio ambiente"]

# Stopwords do português para BoW/TF-IDF (o scikit-learn só traz a lista em inglês).
STOPWORDS_PT = sorted({
    "a", "à", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "às", "até", "com", "como",
    "da", "das", "de", "dela", "delas", "dele", "deles", "depois", "do", "dos", "e", "é", "ela", "elas", "ele",
    "eles", "em", "entre", "era", "eram", "essa", "essas", "esse", "esses", "esta", "está", "estão", "estas",
    "este", "estes", "eu", "foi", "foram", "há", "isso", "isto", "já", "lá", "lhe", "lhes", "mais", "mas", "me",
    "mesmo", "meu", "meus", "minha", "minhas", "muito", "na", "nas", "nem", "no", "nos", "nós", "nossa", "nossas",
    "nosso", "nossos", "num", "numa", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "pra",
    "qual", "quando", "que", "quem", "se", "sem", "ser", "seu", "seus", "só", "sua", "suas", "também", "te",
    "tem", "têm", "tinha", "um", "uma", "uns", "umas", "você", "vocês", "vai", "vão", "então", "onde", "aqui",
    "ali", "cada", "todo", "toda", "todos", "todas", "porque",
})

COR_ALTA, COR_MEDIA = 0.7, 0.5


# ----------------------------------------------------------------------------- dados
def carregar_manifestacoes(caminho: str | Path = PASTA / "manifestacoes.json") -> list[dict]:
    """Lista de manifestações com os campos id, data, categoria_oficial e texto."""
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def carregar_duplicatas_reais(caminho: str | Path = PASTA / "duplicatas_reais.json") -> set[tuple[str, str]]:
    """Gabarito das duplicatas semânticas do conjunto, como pares (id menor, id maior)."""
    pares = json.loads(Path(caminho).read_text(encoding="utf-8"))
    return {tuple(sorted((p["original"], p["duplicata"]))) for p in pares}


# ----------------------------------------------------------------------------- embeddings
class EmbedderFalso:
    """Dublê do SentenceTransformer para testes: hash de trigramas de caracteres em 256 dimensões."""

    dimensao = 256

    def encode(self, textos, normalize_embeddings: bool = True, **_):
        textos = [textos] if isinstance(textos, str) else list(textos)
        matriz = np.zeros((len(textos), self.dimensao), dtype=np.float32)
        for i, texto in enumerate(textos):
            t = f"  {texto.lower()}  "
            for k in range(len(t) - 2):
                h = int(hashlib.md5(t[k : k + 3].encode("utf-8")).hexdigest(), 16)
                matriz[i, h % self.dimensao] += 1.0
        if normalize_embeddings:
            normas = np.linalg.norm(matriz, axis=1, keepdims=True)
            matriz = matriz / np.where(normas == 0, 1, normas)
        return matriz


@lru_cache(maxsize=4)
def carregar_modelo(nome: str = MODELO_PADRAO):
    """SentenceTransformer do nome dado (ou o dublê, se OUVIDORIA_EMBEDDER=falso)."""
    if os.environ.get("OUVIDORIA_EMBEDDER") == "falso":
        return EmbedderFalso()
    warnings.filterwarnings("ignore", message="IProgress not found")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(nome)


def gerar_embeddings(textos: list[str], modelo=None) -> np.ndarray:
    """Embeddings normalizados (norma 1), então o produto escalar já é o cosseno."""
    modelo = modelo or carregar_modelo()
    return np.asarray(modelo.encode(list(textos), normalize_embeddings=True), dtype=np.float32)


def matriz_similaridade(embeddings: np.ndarray) -> np.ndarray:
    """Similaridade de cosseno entre todas as linhas (embeddings já normalizados)."""
    return np.clip(embeddings @ embeddings.T, -1.0, 1.0)


# ----------------------------------------------------------------------------- busca e duplicatas
def buscar(consulta: str, embeddings: np.ndarray, modelo=None, top_k: int = 5) -> list[tuple[int, float]]:
    """Índices das top_k manifestações mais similares à consulta, com o cosseno."""
    q = gerar_embeddings([consulta], modelo)[0]
    sims = embeddings @ q
    ordem = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in ordem]


def cor_do_score(score: float) -> str:
    """🟢 acima de 0,7; 🟡 acima de 0,5; 🔴 nos demais (regra do enunciado)."""
    return "🟢" if score > COR_ALTA else "🟡" if score > COR_MEDIA else "🔴"


def detectar_duplicatas(textos: list[str], limiar: float = 0.85, modelo=None) -> list[tuple[int, int, float]]:
    """Todos os pares (i, j, similaridade), com i < j, cuja similaridade de cosseno passa do limiar.

    Os embeddings são os do modelo multilíngue (ou do `modelo` informado). A lista sai ordenada da
    maior para a menor similaridade.
    """
    sims = matriz_similaridade(gerar_embeddings(textos, modelo))
    pares = [
        (i, j, float(sims[i, j]))
        for i in range(len(textos))
        for j in range(i + 1, len(textos))
        if sims[i, j] > limiar
    ]
    return sorted(pares, key=lambda p: -p[2])


def avaliar_deteccao(previstos: set[tuple[str, str]], reais: set[tuple[str, str]]) -> dict:
    """Verdadeiros positivos, falsos positivos, falsos negativos, precisão, revocação e F1."""
    vp, fp, fn = previstos & reais, previstos - reais, reais - previstos
    precisao = len(vp) / len(previstos) if previstos else 0.0
    revocacao = len(vp) / len(reais) if reais else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
    return {"vp": vp, "fp": fp, "fn": fn, "precisao": precisao, "revocacao": revocacao, "f1": f1}


# ----------------------------------------------------------------------------- chunking
def dividir_em_chunks(texto: str, chunk_size: int, chunk_overlap: int, estrategia: str = "recursiva") -> list[str]:
    """Divide o texto com os splitters do LangChain.

    "recursiva": RecursiveCharacterTextSplitter (parágrafo → linha → frase → palavra → caractere).
    "fixa": CharacterTextSplitter separando só por espaço (tamanho fixo).
    """
    from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    if estrategia == "recursiva":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator="end",
        )
    elif estrategia == "fixa":
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=" ")
    else:
        raise ValueError(f"estratégia desconhecida: {estrategia}")
    return splitter.split_text(texto)
