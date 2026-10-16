# Relatório — análise e transformação de textos

## Objetivo e escopo

O sistema estima a faixa escolar associada a um texto, gera uma variante por substituição lexical e compara as previsões antes e depois da alteração. A classificação não mede diretamente a aprendizagem nem a capacidade de leitura de um estudante. Esta versão implementa a classificação e a troca de palavras; troca de gênero, paráfrase e geração de narrativas não foram implementadas.

## Dados e preparação

O dataset contém 2.076 arquivos. A auditoria encontrou 24 arquivos com rótulos conflitantes e quatro duplicatas adicionais; não encontrou arquivos vazios. Restaram 2.048 documentos: 294 do Ensino Fundamental I, 324 do Ensino Fundamental II, 616 do Ensino Médio e 814 do Ensino Superior. Os arquivos foram normalizados em Unicode NFC, e marcadores `formula_N` foram removidos.

Os documentos foram divididos de forma estratificada em treino, validação e teste antes do treinamento. Nenhum documento ou hash de texto aparece em mais de uma divisão.

| Divisão | Documentos |
|---|---:|
| Treino | 1.433 |
| Validação | 307 |
| Teste | 308 |

O classificador principal recebe um documento completo por exemplo. Cada documento conserva o rótulo da pasta de origem. Esse rótulo representa a classificação existente no dataset e não uma nova anotação especializada do nível de leitura do texto.

## Representação e seleção do modelo

O pipeline selecionado combina TF-IDF de palavras e pares de palavras com TF-IDF de sequências de caracteres de três a cinco posições. A validação mostrou resultado melhor com essa representação do que com a representação anterior, que incluía seis medidas numéricas de texto; por isso, essas medidas não entram no novo classificador. Elas continuam disponíveis na interface como diagnóstico separado.

Regressão logística e SVM linear foram comparadas pela F1 macro na validação, usando documentos completos. Para comparar com a implementação anterior, o treinamento também recriou o método de trechos de até 50 palavras e agregou as previsões por votação, medindo os dois métodos nos mesmos documentos. O SVM com documentos completos foi selecionado.

| Método na validação | Acurácia | F1 macro |
|---|---:|---:|
| SVM, documento completo (palavras + caracteres) | 94,79% | 0,9383 |
| Regressão logística, documento completo (palavras + caracteres) | 93,49% | 0,9221 |
| SVM, votação de trechos (representação anterior) | 89,90% | 0,8770 |
| Regressão logística, votação de trechos (representação anterior) | 89,25% | 0,8767 |
| Referência: classe mais frequente | 39,74% | 0,1422 |

## Avaliação no teste

O teste reservado contém 308 documentos. A comparação abaixo usa exatamente os mesmos documentos e rótulos para os dois métodos:

| Método | Acurácia | F1 macro |
|---|---:|---:|
| Documento completo (palavras + caracteres) | 93,51% | 0,9163 |
| Método anterior: votação de trechos | 88,64% | 0,8634 |

O modelo de documento completo teve resultado melhor neste teste. O relatório detalhado do modelo selecionado é:

| Classe | Documentos | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|
| Ensino Fundamental I | 44 | 89% | 93% | 0,91 |
| Ensino Fundamental II | 48 | 93% | 77% | 0,84 |
| Ensino Médio | 93 | 91% | 97% | 0,94 |
| Ensino Superior | 123 | 98% | 98% | 0,98 |

A F1 ponderada é 0,9339. O recall de 77% para Ensino Fundamental II mostra que essa classe ainda é confundida com as vizinhas. As métricas medem a reprodução dos rótulos herdados das pastas; não comprovam, por si só, que um especialista concordaria com cada classificação.

Dois textos externos enviados como exemplos, um esperado como Ensino Médio (258 palavras) e outro como Ensino Superior (329 palavras), foram avaliados sem serem incorporados ao treinamento. O novo modelo classificou ambos como **Ensino Fundamental II**. A troca de representação melhorou o teste interno, mas não resolveu esses casos externos: o corpus CAEd não representa suficientemente o estilo desses textos. O modelo não usa a simples menção ao nome de uma etapa como regra de classificação.

## Mutações

O dicionário manual contém 486 formas de origem, incluindo flexões de gênero, número e alguns verbos conjugados. Ele inclui, por exemplo, `feliz/alegre`, `bonito/belo`, `cansado/fatigado`, `bom/ótimo` e `longo/extenso`. Uma regra contextual trata `calma` como adjetivo em construções como “a menina estava muito calma”; como substantivo em “mantenha a calma”, ela não é alterada. Na interface, o limite padrão é cinco palavras, configurável até dez; capitalização e pontuação são preservadas.

Na amostra reproduzível de 100 documentos da validação, todos receberam ao menos uma substituição. Nenhuma das variantes alteradas mudou a classe estimada. O experimento permite até duas trocas por documento. Cobertura e mudança de classe são medições do vocabulário e das previsões do modelo; não medem correção semântica ou gramatical. Revise as variantes, pois uma substituição pode alterar a nuance.

## Limitações e próximos passos

Os rótulos vêm das pastas do dataset e não de uma avaliação independente de cada texto por especialistas. A melhora no teste mede documentos do mesmo corpus; não garante acerto em redações externas ou textos gerados por IA. Os dois exemplos externos ainda foram classificados incorretamente, o que aponta para a necessidade de dados representativos do uso esperado, revisados por pessoas que definam o nível. O vocabulário de mutações não cobre qualquer texto; as regras não reconhecem todas as funções gramaticais. A heurística de canonicidade também é aproximada.

Próximos passos úteis são obter exemplos curtos rotulados por especialistas, revisar especialmente os erros de Fundamental II e Ensino Médio, e avaliar as mutações quanto à preservação de sentido e concordância.

## Reprodutibilidade

`treinar.py` é a implementação canônica. Ele audita e divide o dataset, treina com documentos completos, compara com a estratégia anterior em nível de documento, avalia no teste e grava o modelo em `modelos/modelo.joblib` e as evidências em `resultados/`. Execute `python treinar.py` na raiz do projeto para reproduzir o treinamento; a semente padrão é 42. `treino.ipynb` apresenta as etapas e os resultados do mesmo pipeline.
