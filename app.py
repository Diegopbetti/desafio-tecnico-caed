from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from processamento import (
    extrair_caracteristicas,
    tokenizar,
)
from mutacoes import trocar_palavras

st.set_page_config(page_title="Análise de textos — CAEd", layout="wide")
st.title("Análise e transformação de textos")
st.caption(
    "O nível previsto é uma estimativa baseada no dataset escolar. "
    "A classificação considera o texto enviado por inteiro. "
    "A variante troca palavras do vocabulário conhecido ou das regras próprias."
)

CAMINHO_MODELO = Path(__file__).resolve().parent / "modelos" / "modelo.joblib"
if not CAMINHO_MODELO.exists():
    st.info("O modelo ainda não foi preparado. Execute a etapa de treinamento do projeto.")
    st.stop()


@st.cache_resource
def carregar_modelo(caminho, versao_arquivo):
    return joblib.load(caminho)["pipeline"]


modelo = carregar_modelo(str(CAMINHO_MODELO), CAMINHO_MODELO.stat().st_mtime_ns)
arquivo_txt = st.file_uploader("Ou selecione um arquivo .txt", type=["txt"])
with st.form("analise"):
    if arquivo_txt is None:
        texto_digitado = st.text_area(
            "Digite um texto em português",
            value="A menina estava feliz. O menino estava calmo.",
            height=180,
        )
    else:
        texto_digitado = ""
        st.caption(f"Arquivo selecionado: {arquivo_txt.name}")
    limite = st.slider("Máximo de palavras alteradas", 1, 10, 5)
    semente = st.number_input("Semente para repetir a mesma variante", min_value=0, value=42)
    executar = st.form_submit_button("Analisar e gerar variante")

if executar:
    if arquivo_txt is not None:
        conteudo = arquivo_txt.getvalue()
        try:
            texto = conteudo.decode("utf-8-sig")
        except UnicodeDecodeError:
            texto = conteudo.decode("cp1252")
    else:
        texto = texto_digitado
    if not tokenizar(texto):
        st.warning("Digite um texto que contenha palavras.")
        st.stop()
    variante, alteracoes = trocar_palavras(
        texto, limite, int(semente)
    )
    classe_original = modelo.predict([texto])[0]
    classe_variante = modelo.predict([variante])[0]
    esquerda, direita = st.columns(2)
    with esquerda:
        st.subheader("Original")
        st.text(texto)
        st.write("Nível previsto:", classe_original)
    with direita:
        st.subheader("Variante")
        st.text(variante)
        st.write("Nível previsto:", classe_variante)
    if alteracoes:
        st.write("Substituições realizadas")
        st.dataframe(pd.DataFrame(alteracoes)[["original", "nova"]], hide_index=True)
        if len(alteracoes) < limite:
            st.caption(
                f"Foram encontradas {len(alteracoes)} palavra(s) elegível(is) "
                f"para troca, embora o limite esteja em {limite}. Para incluir "
                "mais alterações, aumente o limite."
            )
        st.write("A previsão mudou de nível." if classe_original != classe_variante
                 else "A previsão permaneceu no mesmo nível.")
    else:
        st.info(
            "Não encontrei no texto palavras presentes no dicionário automático."
        )
    st.write("Características dos textos")
    st.dataframe(pd.DataFrame({
        "Original": extrair_caracteristicas(texto),
        "Variante": extrair_caracteristicas(variante),
    }))
