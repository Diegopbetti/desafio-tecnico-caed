from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from processamento import extrair_caracteristicas, tokenizar
from mutacoes import trocar_palavras

st.set_page_config(page_title="Análise de textos — CAEd", layout="wide")
st.title("Análise e transformação de textos")
st.caption("O nível previsto é uma estimativa do modelo baseada no dataset escolar.")

CAMINHO_MODELO = Path(__file__).resolve().parent / "modelos" / "modelo.joblib"
if not CAMINHO_MODELO.exists():
    st.info("O modelo ainda não foi preparado. Execute a etapa de treinamento do projeto.")
    st.stop()


@st.cache_resource
def carregar_modelo(caminho, versao_arquivo):
    return joblib.load(caminho)["pipeline"]


modelo = carregar_modelo(str(CAMINHO_MODELO), CAMINHO_MODELO.stat().st_mtime_ns)
with st.form("analise"):
    texto = st.text_area(
        "Digite um texto em português",
        value="A menina estava feliz. O menino estava calmo.",
        height=180,
    )
    limite = st.slider("Máximo de palavras alteradas", 1, 5, 2)
    semente = st.number_input("Semente para repetir a mesma variante", min_value=0, value=42)
    executar = st.form_submit_button("Analisar e gerar variante")

if executar:
    if not tokenizar(texto):
        st.warning("Digite um texto que contenha palavras.")
        st.stop()
    variante, alteracoes = trocar_palavras(texto, limite, int(semente))
    classe_original, classe_variante = modelo.predict([texto, variante])
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
        st.write("A previsão mudou de nível." if classe_original != classe_variante
                 else "A previsão permaneceu no mesmo nível.")
    else:
        st.info("Nenhuma palavra possui substituição no vocabulário atual.")
    st.write("Características dos textos")
    st.dataframe(pd.DataFrame({
        "Original": extrair_caracteristicas(texto),
        "Variante": extrair_caracteristicas(variante),
    }))