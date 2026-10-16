# Análise e transformação de textos — CAEd

Aplicação Streamlit que estima a faixa escolar associada a um texto e mostra uma variante criada por substituições lexicais. A classificação é uma estimativa baseada no dataset, não uma avaliação individual de leitura.

Esta versão treina e prevê com o documento completo. Implementa substituição de palavras pelo dicionário; não implementa troca de gênero, paráfrase nem criação de narrativas do zero.

## Requisitos e execução no Windows

- Python 3.13
- Dataset em `dataset_caed/`, organizado por faixa de ensino

Na pasta do projeto, abra o PowerShell e execute:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe treinar.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Abra no navegador o endereço local informado pelo Streamlit. Para encerrar, pressione `Ctrl+C` no terminal.

O treinamento separa documentos antes de qualquer modelagem e compara regressão logística e SVM linear por F1 macro na validação. O modelo selecionado usa TF-IDF de palavras e de sequências de caracteres e prevê o documento completo. A comparação com a estratégia anterior de votação entre trechos é mantida como referência. O teste reservado subiu de 92,21% para 93,51% de acurácia e de 0,8944 para 0,9163 de F1 macro. Reexecutar `treinar.py` atualiza o modelo e os resultados.

O notebook `treino.ipynb` apresenta as etapas e os resultados do mesmo treinamento. No VS Code, selecione o kernel da `.venv` e execute as células em ordem.

## Estrutura

- `app.py`: interface web; prevê uma vez para o texto original completo e uma vez para a variante.
- `treinar.py`: auditoria, divisão dos documentos, comparação de métodos, avaliação e exportação do modelo.
- `processamento.py`: limpeza, tokenização e características do texto, incluindo a heurística de canonicidade.
- `mutacoes.py`: dicionário manual de substituições e verificações locais de contexto.
- `treino.ipynb`: apresentação interativa das etapas e dos resultados.
- `dataset_caed/`: textos rotulados pelas quatro pastas de ensino.
- `modelos/`: pipeline de classificação serializado e seus metadados.
- `resultados/`: auditorias, divisão, manifesto dos documentos, comparações, métricas e amostra de mutações.
- `tests/`: testes de processamento, treinamento e mutações.

Os rótulos são herdados das pastas de origem; não há uma anotação independente de especialistas para cada documento. A melhora no teste interno não se repetiu em dois textos externos enviados como exemplos: ambos ainda foram previstos como Fundamental II e não foram usados no treinamento. Para mais números e limitações, consulte `relatorio.md`.
