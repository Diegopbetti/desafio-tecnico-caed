import random
import unicodedata
from processamento import PADRAO_PALAVRA

SINONIMOS = {
    "feliz": ["contente", "alegre"],
    "felizes": ["contentes", "alegres"],
    "bonito": ["belo"], "bonita": ["bela"],
    "bonitos": ["belos"], "bonitas": ["belas"],
    "calmo": ["tranquilo"], "calma": ["tranquila"],
    "calmos": ["tranquilos"], "calmas": ["tranquilas"],
}


def trocar_palavras(texto, limite=2, semente=42):
    texto = unicodedata.normalize("NFC", texto)
    rng = random.Random(semente)
    candidatos = [m for m in PADRAO_PALAVRA.finditer(texto)
                  if m.group().lower() in SINONIMOS]
    quantidade = min(max(0, int(limite)), len(candidatos))
    escolhidos = sorted(rng.sample(candidatos, quantidade), key=lambda m: m.start())
    partes, alteracoes, cursor = [], [], 0
    for trecho in escolhidos:
        original = trecho.group()
        nova = rng.choice(SINONIMOS[original.lower()])
        if original.isupper():
            nova = nova.upper()
        elif original.istitle():
            nova = nova.capitalize()
        partes.extend([texto[cursor:trecho.start()], nova])
        alteracoes.append({"original": original, "nova": nova, "inicio": trecho.start()})
        cursor = trecho.end()
    partes.append(texto[cursor:])
    return "".join(partes), alteracoes