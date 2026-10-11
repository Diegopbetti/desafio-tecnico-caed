import unicodedata
import pytest
from processamento import eh_canonica, extrair_caracteristicas, limpar_texto


@pytest.mark.parametrize("palavra", ["cabeça", "canela", "janela", "luta", "sopa"])
def test_exemplos_canonicos(palavra):
    assert eh_canonica(palavra)


@pytest.mark.parametrize("palavra", ["abajur", "lápis", "cansado", "asa", "alheio"])
def test_exemplos_nao_canonicos(palavra):
    assert not eh_canonica(palavra)


def test_acentos_e_maiusculas():
    assert eh_canonica(unicodedata.normalize("NFD", "CABEÇA"))


def test_texto_sem_palavras():
    assert all(v == 0 for v in extrair_caracteristicas("123 !!!").values())


def test_proporcao_canonicas():
    assert extrair_caracteristicas("Canela, lápis.")["proporcao_canonicas"] == 0.5


def test_marcador_nao_vira_vocabulario():
    assert limpar_texto("Uma formula_12 aqui.") == "Uma aqui."