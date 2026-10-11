import re
import unicodedata

CLASSES = [
    "Ensino Fundamental I",
    "Ensino Fundamental II",
    "Ensino Médio",
    "Ensino Superior",
]

ROTULOS_PASTAS = dict(zip(
    ["1_Ensino_Fundamental_I", "2_Ensino_Fundamental_II",
     "3_Ensino_Medio", "4_Ensino_Superior"],
    CLASSES,
))

PADRAO_PALAVRA = re.compile(r"[^\W\d_]+", re.UNICODE)
PADRAO_CV = re.compile(
    r"(?:ch|lh|nh|rr|ss|[bcçdfgjklmnprstvz])[aeiouáéíóúâêôàãõ]"
)

NOMES_FEATURES = [
    "n_palavras", "n_frases", "palavras_por_frase",
    "letras_por_palavra", "proporcao_unicas", "proporcao_canonicas",
]


def limpar_texto(texto):
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r"\bformula_\d+\b", " ", texto, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", texto).strip()


def tokenizar(texto):
    return PADRAO_PALAVRA.findall(limpar_texto(texto))


def eh_canonica(palavra):
    palavra = unicodedata.normalize("NFC", palavra).lower()
    if not palavra or not palavra.isalpha():
        return False
    if re.search(r"[qg]u[eiéêí]", palavra):
        return False
    unidades = PADRAO_CV.findall(palavra)
    return bool(unidades) and "".join(unidades) == palavra


def extrair_caracteristicas(texto):
    limpo = limpar_texto(texto)
    palavras = [p.lower() for p in tokenizar(limpo)]
    n = len(palavras)
    if n == 0:
        return dict.fromkeys(NOMES_FEATURES, 0.0)
    # Separação aproximada; abreviações como "Sr."
    frases = [f for f in re.split(r"[.!?]+", limpo) if tokenizar(f)]
    n_frases = max(1, len(frases))
    return {
        "n_palavras": n,
        "n_frases": n_frases,
        "palavras_por_frase": n / n_frases,
        "letras_por_palavra": sum(map(len, palavras)) / n,
        "proporcao_unicas": len(set(palavras)) / n,
        "proporcao_canonicas": sum(map(eh_canonica, palavras)) / n,
    }


def transformar_textos(textos):
    # Ordem fixa das colunas no treino e na aplicação
    return [
        [valores[nome] for nome in NOMES_FEATURES]
        for valores in map(extrair_caracteristicas, textos)
    ]


def texto_para_tfidf(texto):
    return limpar_texto(texto).lower()