from pathlib import Path

import pandas as pd

from processamento import CLASSES
from treinar import (
    ROTULOS_PASTAS,
    _prever_documentos_por_trechos,
    dividir_documentos,
    executar_treinamento,
    gerar_trechos,
)


def test_divisao_separa_documentos_inteiros_e_preserva_classes(tmp_path: Path):
    documentos = pd.DataFrame([
        {
            "documento": f"{classe}-{i}.txt",
            "classe": classe,
            "hash_texto": f"hash-{classe}-{i}",
            "texto": "Texto de exemplo.",
        }
        for classe in CLASSES
        for i in range(10)
    ])

    partes = dividir_documentos(documentos, tmp_path)

    assert set(partes) == {"treino", "validacao", "teste"}
    conjuntos = {nome: set(tabela.documento) for nome, tabela in partes.items()}
    assert conjuntos["treino"].isdisjoint(conjuntos["validacao"])
    assert conjuntos["treino"].isdisjoint(conjuntos["teste"])
    assert conjuntos["validacao"].isdisjoint(conjuntos["teste"])
    assert all(set(tabela.classe) == set(CLASSES) for tabela in partes.values())


def test_referencia_legada_usa_trechos_limitados_e_distribuidos():
    texto = " ".join(f"palavra{i}" for i in range(160))
    documentos = pd.DataFrame([{
        "documento": "origem.txt",
        "classe": CLASSES[0],
        "texto": texto,
    }])

    trechos = gerar_trechos(documentos)

    assert len(trechos) == 3
    assert trechos.n_palavras.max() <= 50
    assert trechos.numero_trecho.tolist() == [1, 2, 3]
    assert trechos.documento.unique().tolist() == ["origem.txt"]
    assert trechos.classe.unique().tolist() == [CLASSES[0]]


def test_previsao_legada_agrega_votos_em_um_resultado_por_documento():
    class ModeloPorPalavra:
        def predict(self, textos):
            return [CLASSES[0] if texto.startswith("inicio") else CLASSES[1]
                    for texto in textos]

    documentos = pd.DataFrame([{
        "documento": "origem.txt",
        "classe": CLASSES[1],
        "texto": " ".join(["inicio"] + [f"palavra{i}" for i in range(160)]),
    }])

    previsoes = _prever_documentos_por_trechos(ModeloPorPalavra(), documentos)

    assert previsoes == [CLASSES[1]]


def test_treinamento_final_usa_documentos_completos(tmp_path: Path):
    dataset = tmp_path / "dataset"
    for indice_classe, (pasta_rotulada, classe) in enumerate(
        zip(ROTULOS_PASTAS, CLASSES)
    ):
        pasta = dataset / pasta_rotulada
        pasta.mkdir(parents=True)
        for indice_documento in range(12):
            texto = (
                f"texto classe{indice_classe} vocabulario{indice_classe} "
                f"documento{indice_documento} exemplo educacao leitura escrita. "
            ) * 20
            (pasta / f"documento{indice_documento}.txt").write_text(
                texto, encoding="utf-8",
            )

    resultado = executar_treinamento(dataset=dataset, raiz=tmp_path)

    assert resultado["modelo"]["unidade_treino"] == "documento completo"
    assert resultado["comparacao_teste"]["modelo"].tolist() == [
        "documento_completo", "metodo_anterior_trechos",
    ]
    for nome, tabela in resultado["documentos"].items():
        assert len(tabela) > 0, nome
