from mutacoes import trocar_palavras


def test_sem_candidato():
    assert trocar_palavras("A menina leu.") == ("A menina leu.", [])


def test_reproduzivel_e_preserva_pontuacao():
    texto = "Bonita, calma e feliz!"
    primeira = trocar_palavras(texto, limite=3, semente=7)
    assert primeira == trocar_palavras(texto, limite=3, semente=7)
    assert primeira[0].startswith("Bela,")
    assert primeira[0].endswith("!")
    assert len(primeira[1]) == 3


def test_nao_substitui_parte_de_palavra():
    assert trocar_palavras("infeliz") == ("infeliz", [])


def test_limite_zero():
    assert trocar_palavras("Ela está feliz.", limite=0) == ("Ela está feliz.", [])