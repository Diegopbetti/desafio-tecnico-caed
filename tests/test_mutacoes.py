from mutacoes import trocar_palavras


def test_sem_candidato():
    assert trocar_palavras("A menina leu.") == ("A menina leu.", [])


def test_reproduzivel_e_preserva_pontuacao():
    texto = "Bonita, lenta e feliz!"
    primeira = trocar_palavras(texto, limite=3, semente=7)
    assert primeira == trocar_palavras(texto, limite=3, semente=7)
    assert primeira[0].startswith("Bela,")
    assert primeira[0].endswith("!")
    assert len(primeira[1]) == 3


def test_nao_substitui_parte_de_palavra():
    assert trocar_palavras("infeliz") == ("infeliz", [])


def test_limite_zero():
    assert trocar_palavras("Ela está feliz.", limite=0) == ("Ela está feliz.", [])


def test_vocabulario_ampliado_substitui_varias_formas():
    texto = "A menina ficou cansada, feliz e importante."
    variante, alteracoes = trocar_palavras(texto, limite=3)
    assert len(alteracoes) == 3
    assert "fatigada" in variante
    assert "alegre" in variante or "contente" in variante
    assert "relevante" in variante


def test_vocabulario_cobre_qualidades_e_tamanhos_com_concordancia():
    variante, alteracoes = trocar_palavras(
        "Uma boa história antiga e um texto longo.", limite=3,
    )
    assert len(alteracoes) == 3
    assert "ótima" in variante
    assert "velha" in variante
    assert "extenso" in variante


def test_calma_substantivo_nao_e_substituido():
    assert trocar_palavras("Ela pediu que todos mantivessem a calma.") == (
        "Ela pediu que todos mantivessem a calma.", []
    )


def test_calma_adjetivo_e_substituida_com_concordancia():
    variante, alteracoes = trocar_palavras("A menina estava muito calma.")
    assert variante == "A menina estava muito tranquila."
    assert alteracoes[0]["original"] == "calma"
    assert alteracoes[0]["nova"] == "tranquila"


def test_preserva_caixa_e_pontuacao_com_vocabulario_ampliado():
    variante, alteracoes = trocar_palavras("IMPORTANTE!", limite=1)
    assert len(alteracoes) == 1
    assert alteracoes[0]["nova"].isupper()
    assert variante.endswith("!")


def test_texto_com_vocabulario_comum_tambem_muda():
    variante, alteracoes = trocar_palavras(
        "O projeto pequeno começou e ajudou muitas pessoas.", limite=3,
    )
    assert len(alteracoes) == 3
    assert "miúdo" in variante
    assert "iniciou" in variante
    assert "auxiliou" in variante


def test_vocabulario_automatico_cobre_formas_comuns_sem_regra_manual():
    texto = "Alunos estudam em escolas novas e tentam entender temas difíceis."
    variante, alteracoes = trocar_palavras(texto, limite=10)

    assert len(alteracoes) == 6
    assert "Estudantes" in variante
    assert "instituições de ensino" in variante
    assert "recentes" in variante
    assert "compreender" in variante
    assert "assuntos" in variante
    assert "nada fáceis" in variante
