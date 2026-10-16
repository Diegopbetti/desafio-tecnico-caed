"""Audita o dataset, treina classificadores em documentos e salva resultados."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from processamento import (
    CLASSES,
    ROTULOS_PASTAS,
    extrair_caracteristicas,
    limpar_texto,
    tokenizar,
    texto_para_tfidf,
    transformar_textos,
)
from mutacoes import trocar_palavras

SEMENTE = 42
PALAVRAS_POR_TRECHO = 50
TRECHOS_POR_DOCUMENTO = 3
AMOSTRA_EXPERIMENTO_MUTACOES = 100
RAIZ = Path(__file__).resolve().parent


def carregar_e_auditar(dataset, resultados):
    """Lê os documentos, remove vazios, duplicatas e conflitos entre classes."""
    registros = []
    for arquivo in sorted(dataset.rglob("*.txt")):
        pasta = arquivo.parent.name
        if pasta not in ROTULOS_PASTAS:
            raise ValueError(f"Pasta sem rótulo de classe conhecido: {pasta}")
        texto = limpar_texto(arquivo.read_text(encoding="utf-8-sig"))
        registros.append({
            "documento": arquivo.relative_to(dataset).as_posix(),
            "texto": texto,
            "classe": ROTULOS_PASTAS[pasta],
            "n_palavras": len(tokenizar(texto)),
            "hash_texto": hashlib.sha256(texto.lower().encode("utf-8")).hexdigest(),
        })

    df = pd.DataFrame(registros)
    if df.empty:
        raise ValueError(f"Nenhum .txt foi encontrado em {dataset}")

    print(f"Arquivos originais: {len(df)}")
    print(df.groupby("classe")["n_palavras"].agg(["count", "min", "median", "max"]))

    vazios = df[df["n_palavras"] == 0].copy()
    df = df[df["n_palavras"] > 0].copy()
    rotulos_por_hash = df.groupby("hash_texto")["classe"].nunique()
    hashes_conflitantes = rotulos_por_hash[rotulos_por_hash > 1].index
    conflitos = df[df["hash_texto"].isin(hashes_conflitantes)].copy()
    df = df[~df["hash_texto"].isin(hashes_conflitantes)].copy()
    duplicatas = df[df.duplicated("hash_texto", keep="first")].copy()
    df = df.drop_duplicates("hash_texto").reset_index(drop=True)

    resultados.mkdir(parents=True, exist_ok=True)
    for nome, tabela in (("vazios", vazios), ("conflitos", conflitos),
                         ("duplicatas", duplicatas)):
        tabela.drop(columns="texto").to_csv(resultados / f"{nome}.csv", index=False)
        print(f"{nome}: {len(tabela)} arquivo(s)")

    if set(df["classe"]) != set(CLASSES):
        raise ValueError("O dataset limpo não contém as quatro classes esperadas.")
    print(f"Documentos válidos após a auditoria: {len(df)}")
    return df


def dividir_documentos(df, resultados, semente=SEMENTE):
    """Separa documentos completos antes de qualquer representação dos textos."""
    treino, restante = train_test_split(
        df, test_size=0.30, stratify=df["classe"], random_state=semente,
    )
    validacao, teste = train_test_split(
        restante, test_size=0.50, stratify=restante["classe"], random_state=semente,
    )
    partes = {"treino": treino, "validacao": validacao, "teste": teste}

    for nome_a, nome_b in (("treino", "validacao"), ("treino", "teste"),
                           ("validacao", "teste")):
        docs_a = set(partes[nome_a]["documento"])
        docs_b = set(partes[nome_b]["documento"])
        if not docs_a.isdisjoint(docs_b):
            raise AssertionError(f"Documentos aparecem em {nome_a} e {nome_b}.")

    manifesto = pd.concat(
        tabela[["documento", "classe", "hash_texto"]].assign(conjunto=nome)
        for nome, tabela in partes.items()
    ).reset_index(drop=True)
    manifesto.to_csv(resultados / "divisao_dados.csv", index=False)
    print("Divisão por documento:")
    print(pd.crosstab(manifesto["classe"], manifesto["conjunto"]))
    return partes


def gerar_trechos(documentos, tamanho=PALAVRAS_POR_TRECHO,
                  quantidade=TRECHOS_POR_DOCUMENTO):
    """Cria até três trechos distribuídos ao longo de cada documento."""
    trechos = []
    for linha in documentos.itertuples(index=False):
        palavras = linha.texto.split()
        n_trechos = min(quantidade, max(1, len(palavras) // tamanho))
        margem = max(0, len(palavras) - tamanho)
        if n_trechos == 1:
            inicios = [0]
        else:
            inicios = [round(i * margem / (n_trechos - 1)) for i in range(n_trechos)]
        for numero, inicio in enumerate(inicios, start=1):
            trecho = " ".join(palavras[inicio:inicio + tamanho])
            trechos.append({
                "documento": linha.documento,
                "classe": linha.classe,
                "trecho": trecho,
                "n_palavras": len(tokenizar(trecho)),
                "numero_trecho": numero,
            })
    return pd.DataFrame(trechos)


def criar_representacao():
    """Combina vocabulário e padrões ortográficos de palavras completas."""
    return FeatureUnion([
        ("palavras", TfidfVectorizer(
            preprocessor=texto_para_tfidf,
            ngram_range=(1, 2), min_df=2, max_features=30000,
            sublinear_tf=True,
        )),
        ("caracteres", TfidfVectorizer(
            preprocessor=texto_para_tfidf,
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            max_features=50000, sublinear_tf=True,
        )),
    ])


def criar_representacao_legada():
    """Mantém a representação anterior para comparar o método por trechos."""
    return FeatureUnion([
        ("vocabulario", TfidfVectorizer(
            preprocessor=texto_para_tfidf,
            ngram_range=(1, 2), min_df=2, max_features=20000,
        )),
        ("numericas", Pipeline([
            ("extrair", FunctionTransformer(transformar_textos, validate=False)),
            ("padronizar", StandardScaler()),
        ])),
    ])


def criar_modelos(semente=SEMENTE, fabrica_representacao=criar_representacao):
    return {
        "logistica": Pipeline([
            ("representacao", fabrica_representacao()),
            ("classificador", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=semente,
            )),
        ]),
        "svm_linear": Pipeline([
            ("representacao", fabrica_representacao()),
            ("classificador", LinearSVC(
                class_weight="balanced", random_state=semente, max_iter=10000,
            )),
        ]),
    }


def _prever_documentos_por_trechos(modelo, documentos):
    """Reproduz a estratégia anterior: prevê trechos e vota por documento."""
    trechos = gerar_trechos(documentos)
    if trechos.empty:
        return []
    trechos["previsao"] = modelo.predict(trechos["trecho"])
    votos = {}
    for documento, linhas in trechos.groupby("documento", sort=False):
        votos[documento] = Counter(linhas["previsao"]).most_common(1)[0][0]
    return documentos["documento"].map(votos).tolist()


def _linha_metricas(nome, rotulos, previsoes, unidade):
    return {
        "modelo": nome,
        "unidade": unidade,
        "acuracia": accuracy_score(rotulos, previsoes),
        "f1_macro": f1_score(
            rotulos, previsoes, average="macro", zero_division=0,
        ),
        "f1_ponderado": f1_score(
            rotulos, previsoes, average="weighted", zero_division=0,
        ),
    }


def _experimento_mutacoes(modelo, textos, resultados, semente=SEMENTE):
    amostra = textos.sample(
        min(AMOSTRA_EXPERIMENTO_MUTACOES, len(textos)), random_state=semente,
    )
    entradas = [("demonstracao", "A menina estava feliz. O menino estava calmo.")]
    entradas.extend(
        (linha.documento, linha.texto)
        for linha in amostra.itertuples(index=False)
    )

    experimentos = []
    for origem, texto in entradas:
        variante, alteracoes = trocar_palavras(texto, limite=2, semente=semente)
        antes, depois = modelo.predict([texto, variante])
        f_antes = extrair_caracteristicas(texto)
        f_depois = extrair_caracteristicas(variante)
        experimentos.append({
            "origem": origem,
            "original": texto,
            "variante": variante,
            "classe_antes": antes,
            "classe_depois": depois,
            "n_alteracoes": len(alteracoes),
            "mudou_classe": bool(antes != depois),
            "canonicas_antes": f_antes["proporcao_canonicas"],
            "canonicas_depois": f_depois["proporcao_canonicas"],
            "alteracoes": json.dumps(alteracoes, ensure_ascii=False),
        })

    tabela = pd.DataFrame(experimentos)
    tabela.to_csv(resultados / "experimentos_mutacao.csv", index=False)
    reais = tabela[tabela["origem"] != "demonstracao"]
    alterados = reais[reais["n_alteracoes"] > 0]
    cobertura = len(alterados) / len(reais) if len(reais) else 0.0
    proporcao_mudanca = (
        float(alterados["mudou_classe"].mean()) if len(alterados) else 0.0
    )
    print(f"Cobertura das mutações: {len(alterados)}/{len(reais)} ({cobertura:.1%})")
    if len(alterados):
        print(f"Mudança de nível entre documentos alterados: {proporcao_mudanca:.1%}")
    return tabela, cobertura, proporcao_mudanca


def executar_treinamento(dataset=None, raiz=None, semente=SEMENTE):
    raiz = Path(raiz) if raiz else RAIZ
    dataset = Path(dataset) if dataset else raiz / "dataset_caed"
    if not dataset.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset}")
    diretorio_resultados = raiz / "resultados"
    diretorio_modelos = raiz / "modelos"

    df = carregar_e_auditar(dataset, diretorio_resultados)
    partes = dividir_documentos(df, diretorio_resultados, semente)
    manifesto_documentos = pd.concat([
        tabela[["documento", "classe", "hash_texto", "n_palavras"]]
        .assign(conjunto=nome)
        for nome, tabela in partes.items()
    ], ignore_index=True)
    manifesto_documentos.to_csv(
        diretorio_resultados / "manifesto_documentos.csv", index=False,
    )
    for nome, tabela in partes.items():
        print(f"Documentos de {nome}: {len(tabela)}")

    rotulos_validacao = partes["validacao"]["classe"]
    referencia = DummyClassifier(strategy="most_frequent")
    referencia.fit([[0]] * len(partes["treino"]), partes["treino"]["classe"])
    pred_ref = referencia.predict([[0]] * len(partes["validacao"]))
    comparacoes = [_linha_metricas(
        "referencia_mais_frequente", rotulos_validacao, pred_ref, "documento",
    )]

    modelos_documentos = criar_modelos(semente)
    for nome, modelo in modelos_documentos.items():
        print(f"Treinando documentos completos: {nome}...", flush=True)
        modelo.fit(partes["treino"]["texto"], partes["treino"]["classe"])
        previsto = modelo.predict(partes["validacao"]["texto"])
        comparacoes.append(_linha_metricas(
            f"documento_completo_{nome}", rotulos_validacao, previsto,
            "documento completo",
        ))
        print(classification_report(
            rotulos_validacao, previsto, labels=CLASSES, zero_division=0,
        ))

    # Recria a abordagem anterior para uma comparação pareada por documento.
    trechos_treino_legado = gerar_trechos(partes["treino"])
    modelos_legados = criar_modelos(
        semente, fabrica_representacao=criar_representacao_legada,
    )
    for nome, modelo in modelos_legados.items():
        print(f"Treinando referência por trechos: {nome}...", flush=True)
        modelo.fit(trechos_treino_legado["trecho"], trechos_treino_legado["classe"])
        previsto = _prever_documentos_por_trechos(modelo, partes["validacao"])
        comparacoes.append(_linha_metricas(
            f"metodo_anterior_{nome}", rotulos_validacao, previsto,
            "votação de até 3 trechos/documento",
        ))

    comparacao = pd.DataFrame(comparacoes).sort_values(
        "f1_macro", ascending=False,
    ).reset_index(drop=True)
    comparacao.to_csv(
        diretorio_resultados / "comparacao_validacao.csv", index=False,
    )
    print("Comparação na validação (um resultado por documento):")
    print(comparacao.to_string(index=False))
    comparacao_documentos = comparacao[
        comparacao["modelo"].str.startswith("documento_completo_")
    ]
    melhor_nome = comparacao_documentos.iloc[0]["modelo"].removeprefix(
        "documento_completo_"
    )
    melhor_modelo = modelos_documentos[melhor_nome]
    comparacao_legada = comparacao[
        comparacao["modelo"].str.startswith("metodo_anterior_")
    ]
    melhor_legado_nome = comparacao_legada.iloc[0]["modelo"].removeprefix(
        "metodo_anterior_"
    )
    melhor_modelo_legado = modelos_legados[melhor_legado_nome]

    teste = partes["teste"]
    previsto_teste = melhor_modelo.predict(teste["texto"])
    previsto_teste_legado = _prever_documentos_por_trechos(
        melhor_modelo_legado, teste,
    )
    avaliacao = classification_report(
        teste["classe"], previsto_teste, labels=CLASSES,
        output_dict=True, zero_division=0,
    )
    (diretorio_resultados / "metricas_teste.json").write_text(
        json.dumps(avaliacao, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    matriz = pd.DataFrame(
        confusion_matrix(teste["classe"], previsto_teste, labels=CLASSES),
        index=CLASSES, columns=CLASSES,
    )
    matriz.to_csv(
        diretorio_resultados / "matriz_confusao.csv", index_label="real",
    )
    comparacao_teste = pd.DataFrame([
        _linha_metricas(
            "documento_completo", teste["classe"], previsto_teste,
            "documento completo",
        ),
        _linha_metricas(
            "metodo_anterior_trechos", teste["classe"], previsto_teste_legado,
            "votação de até 3 trechos/documento",
        ),
    ])
    comparacao_teste.to_csv(
        diretorio_resultados / "comparacao_teste.csv", index=False,
    )
    print("Avaliação final no teste (documentos completos):")
    print(classification_report(
        teste["classe"], previsto_teste, labels=CLASSES, zero_division=0,
    ))
    print("Comparação no teste (mesmos documentos):")
    print(comparacao_teste.to_string(index=False))

    diretorio_modelos.mkdir(parents=True, exist_ok=True)
    pacote = {
        "pipeline": melhor_modelo,
        "nome": melhor_nome,
        "classes": CLASSES,
        "semente": semente,
        "sklearn_version": sklearn.__version__,
        "unidade_treino": "documento completo",
        "rotulo_documento": "classe da pasta de origem",
        "representacao": "TF-IDF de palavras (1-2) e caracteres (3-5)",
    }
    caminho_modelo = diretorio_modelos / "modelo.joblib"
    joblib.dump(pacote, caminho_modelo)
    carregado = joblib.load(caminho_modelo)["pipeline"]
    amostra_teste = teste["texto"].iloc[0]
    if carregado.predict([amostra_teste])[0] != melhor_modelo.predict([amostra_teste])[0]:
        raise AssertionError("O modelo recarregado diverge do modelo treinado.")

    experimentos, cobertura, proporcao_mudanca = _experimento_mutacoes(
        melhor_modelo, partes["validacao"], diretorio_resultados, semente,
    )
    return {
        "comparacao": comparacao,
        "comparacao_teste": comparacao_teste,
        "metricas_teste": avaliacao,
        "matriz_confusao": matriz,
        "experimentos_mutacao": experimentos,
        "cobertura_mutacoes": cobertura,
        "mudanca_classe_mutacoes": proporcao_mudanca,
        "documentos": partes,
        "modelo": pacote,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Audita o dataset, compara modelos e salva o modelo CAEd.",
    )
    parser.add_argument(
        "--dataset", type=Path, default=RAIZ / "dataset_caed",
        help="Pasta do dataset organizado por classe (padrão: ./dataset_caed).",
    )
    parser.add_argument(
        "--raiz", type=Path, default=RAIZ,
        help="Pasta para modelos/ e resultados/ (padrão: pasta deste arquivo).",
    )
    parser.add_argument("--semente", type=int, default=SEMENTE)
    args = parser.parse_args(argv)
    executar_treinamento(args.dataset, args.raiz, args.semente)


if __name__ == "__main__":
    main()
