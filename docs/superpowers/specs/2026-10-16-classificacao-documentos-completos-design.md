# Classificação de documentos completos

## Objetivo

Treinar e usar o classificador sobre documentos completos, em vez de juntar previsões de trechos de até 50 palavras. A aplicação deve produzir uma classificação única para o texto inteiro. O texto fornecido pelo usuário, esperado como Ensino Médio, será uma verificação externa e não fará parte do treinamento.

## Estado atual

O treinamento divide cada documento em até três trechos distribuídos de 50 palavras, treina modelos nesses trechos e avalia as previsões por trecho. O Streamlit repete a divisão e escolhe o nível mais frequente entre as previsões. Assim, documentos curtos e longos são representados por unidades diferentes da classificação que o usuário espera.

## Desenho proposto

- Preservar os rótulos existentes, derivados das quatro pastas do dataset.
- Preservar a divisão estratificada por documento para evitar que partes de um mesmo documento apareçam em conjuntos diferentes.
- Treinar e validar os modelos com um registro por documento completo, comparando regressão logística e SVM linear pelo F1 macro na validação.
- Avaliar a escolha no conjunto de teste por documento completo, registrando acurácia, F1 macro, F1 por classe e matriz de confusão.
- Alterar o Streamlit para chamar o modelo uma vez com o texto completo e mostrar uma única previsão, removendo a votação por trechos.
- Reavaliar o modelo atual em nível de documento (votação das previsões por trecho) para comparar os dois métodos no mesmo conjunto de documentos e com os mesmos rótulos.
- Atualizar o notebook, relatório e resultados gerados para descrever claramente a unidade documental e os resultados medidos.
- Testar o exemplo fornecido como dado externo, sem adicioná-lo ao dataset nem codificar sua classe como regra. A expectativa informada é Ensino Médio; se o modelo discordar, registrar o resultado em vez de forçá-lo.

## Limites e riscos

A mudança alinha a unidade treinada à unidade apresentada pelo usuário, mas não garante maior desempenho nem que o exemplo externo seja classificado como Ensino Médio. Os rótulos continuam sendo os rótulos herdados das pastas do dataset; não há anotação independente de especialistas para todos os documentos. A avaliação comparativa deve deixar explícita essa diferença e não misturar métricas por trecho com métricas por documento.

## Critérios de aceitação

1. O treinamento usa um exemplo por documento completo e continua separando documentos antes de qualquer processamento que possa causar vazamento.
2. A aplicação produz exatamente uma previsão para o texto enviado, sem votação entre trechos.
3. O modelo atual e o modelo de documento completo são comparados no mesmo conjunto de documentos, com métricas por documento.
4. O exemplo fornecido permanece fora do treinamento e seu resultado é reportado sem ajuste manual da classe.
5. Testes, notebook e relatório correspondem ao código e aos novos artefatos.
