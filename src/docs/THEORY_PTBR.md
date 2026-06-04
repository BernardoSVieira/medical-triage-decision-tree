# Teoria e Implementação do Projeto

## Objetivo

O objetivo deste projeto é simular um sistema simples de triagem médica utilizando uma Árvore de Decisão (*Decision Tree Classifier*).

O sistema recebe um conjunto de sintomas informados pelo paciente e classifica o nível de urgência em:

- Baixa
- Média
- Alta

Este projeto foi desenvolvido com fins educacionais, buscando aplicar conceitos de Aprendizado de Máquina (*Machine Learning*) e compreender o funcionamento interno das Árvores de Decisão.

---

# Conjunto de Dados (Dataset)

O dataset utilizado foi criado manualmente e contém registros sintéticos de pacientes.

Cada paciente é representado por seis sintomas binários:

| Sintoma | Descrição |
|----------|------------|
| Febre | O paciente apresenta febre |
| Tosse | O paciente apresenta tosse |
| DorPeito | O paciente apresenta dor no peito |
| FaltaAr | O paciente apresenta dificuldade para respirar |
| DorCabeca | O paciente apresenta dor de cabeça |
| Nausea | O paciente apresenta náusea |

Os valores possíveis são:

- 1 = Sintoma presente
- 0 = Sintoma ausente

A variável alvo (*target*) é:

| Urgência |
|-----------|
| Baixa |
| Média |
| Alta |

---

# Pré-processamento dos Dados

Para que o algoritmo possa trabalhar corretamente, os dados precisam estar em formato numérico.

Como os sintomas já foram representados por valores binários (0 e 1), apenas a coluna de classificação de urgência precisou ser convertida.

Foi utilizado o `LabelEncoder` da biblioteca Scikit-Learn para transformar:

| Valor Original | Valor Codificado |
|----------------|------------------|
| Alta | 0 |
| Baixa | 1 |
| Média | 2 |

> Observação: os números atribuídos podem variar dependendo da ordem em que o LabelEncoder encontra as classes.

---

# Por que utilizar uma Árvore de Decisão?

As Árvores de Decisão são um dos algoritmos mais intuitivos do Machine Learning.

Seu funcionamento se assemelha a um fluxo de perguntas e respostas.

Exemplo simplificado:

```text
Dor no peito?

├── Sim → Alta Urgência
└── Não
    │
    ├── Febre?
    │   ├── Sim → Média Urgência
    │   └── Não → Baixa Urgência
```

O algoritmo aprende automaticamente quais perguntas são mais importantes para separar os casos em diferentes categorias.

---

# Entropia

Neste projeto foi utilizado:

```python
criterion="entropy"
```

A entropia é uma medida de desorganização ou incerteza dos dados.

### Baixa Entropia

Quando todos os exemplos pertencem à mesma classe.

Exemplo:

```text
Alta
Alta
Alta
Alta
```

Nesse caso existe pouca incerteza.

---

### Alta Entropia

Quando diferentes classes estão misturadas.

Exemplo:

```text
Alta
Baixa
Média
Alta
Baixa
```

Nesse caso existe muita incerteza.

---

# Ganho de Informação (Information Gain)

O objetivo da Árvore de Decisão é reduzir a entropia o máximo possível.

Para isso, o algoritmo avalia cada sintoma e verifica qual deles consegue separar melhor os dados.

Essa redução da entropia recebe o nome de **Ganho de Informação**.

O sintoma que produz o maior ganho de informação é escolhido primeiro na árvore.

Por exemplo:

```text
DorPeito
```

pode ser considerado mais importante do que:

```text
Nausea
```

caso consiga separar melhor os pacientes em diferentes níveis de urgência.

---

# Divisão Treino e Teste (Train/Test Split)

Para verificar se o modelo realmente aprendeu padrões úteis, o dataset foi dividido em duas partes:

- 70% para treinamento
- 30% para teste

Utilizando:

```python
train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
```

### Dados de Treino

Utilizados para ensinar o modelo.

### Dados de Teste

Utilizados para avaliar o desempenho em exemplos nunca vistos anteriormente.

Isso evita que o modelo apenas memorize os dados.

---

# Treinamento do Modelo

Após o pré-processamento, foi criado um classificador:

```python
clf = DecisionTreeClassifier(
    criterion="entropy"
)
```

Em seguida o modelo foi treinado:

```python
clf.fit(X_train, y_train)
```

Nesse momento a Árvore de Decisão analisa todos os exemplos de treinamento e constrói automaticamente sua estrutura de decisão.

---

# Avaliação do Modelo

Após o treinamento, é necessário verificar se o modelo está funcionando corretamente.

Para isso foram utilizadas duas métricas principais:

- Accuracy
- Matriz de Confusão

---

## Accuracy (Acurácia)

A Accuracy mede a porcentagem de previsões corretas realizadas pelo modelo.

Fórmula:

```text
Accuracy = Acertos / Total de Casos
```

Exemplo:

```text
23 acertos em 24 casos
```

Resultado:

```text
Accuracy = 95,8%
```

Quanto mais próximo de 100%, melhor o desempenho do modelo.

---

# Matriz de Confusão

A Matriz de Confusão mostra detalhadamente onde o modelo acertou e onde errou.

Exemplo:

```text
[[12 0 0]
 [ 0 4 1]
 [ 0 0 7]]
```

Considerando a ordem:

```text
Alta
Baixa
Média
```

A interpretação seria:

- 12 casos de Alta classificados corretamente
- 4 casos de Baixa classificados corretamente
- 1 caso de Baixa classificado incorretamente como Média
- 7 casos de Média classificados corretamente

A matriz permite identificar exatamente quais classes estão sendo confundidas.

---

# Sistema de Triagem Interativa

Após o treinamento do modelo, o usuário pode realizar uma triagem diretamente pelo terminal.

Exemplo:

```text
Paciente possui Febre? (S/N)
Paciente possui Tosse? (S/N)
Paciente possui DorPeito? (S/N)
Paciente possui FaltaAr? (S/N)
Paciente possui DorCabeca? (S/N)
Paciente possui Nausea? (S/N)
```

As respostas são convertidas para valores binários:

```text
S → 1
N → 0
```

Em seguida é criado um novo registro:

```python
[1, 0, 1, 1, 0, 0]
```

que é enviado ao modelo através do método:

```python
clf.predict()
```

O sistema então retorna o nível de urgência previsto.

---

# Visualização da Árvore

O projeto também permite visualizar a Árvore de Decisão gerada.

Foi utilizada a função:

```python
plot_tree()
```

Essa visualização ajuda a compreender:

- Quais sintomas são mais importantes;
- Como as decisões são tomadas;
- Como os pacientes são classificados em cada nível de urgência.

---

# Limitações do Projeto

Este projeto possui caráter exclusivamente educacional.

O dataset utilizado é sintético e foi criado manualmente para fins de estudo.

Portanto:

- Não deve ser utilizado para diagnósticos médicos reais;
- Não substitui profissionais da saúde;
- Não representa protocolos médicos oficiais.

---

# Conhecimentos Aplicados

Durante o desenvolvimento deste projeto foram aplicados os seguintes conceitos:

- Manipulação de dados com Pandas
- Machine Learning com Scikit-Learn
- Árvores de Decisão
- Entropia
- Ganho de Informação
- Label Encoding
- Train/Test Split
- Accuracy
- Matriz de Confusão
- Visualização de modelos
- Entrada de dados pelo terminal
- Organização de projetos Python
- Controle de versão com Git e GitHub

---

# Aprendizados Obtidos

Durante a construção deste projeto foi possível compreender na prática:

- Como preparar um dataset para treinamento;
- Como uma Árvore de Decisão toma decisões;
- Como avaliar a qualidade de um modelo de Machine Learning;
- Como transformar dados de entrada em previsões;
- Como integrar um modelo treinado com uma aplicação interativa em Python.

O principal objetivo do projeto não foi apenas obter previsões corretas, mas entender todo o processo de construção, treinamento e avaliação de um modelo de Machine Learning.

---

# Referências e Créditos

Este projeto foi desenvolvido como atividade prática de estudo sobre Árvores de Decisão e Machine Learning.

A principal referência utilizada para a implementação inicial e compreensão dos conceitos foi o material disponibilizado pelo professor **Saulo Popov Zambiasi**, por meio da Wiki ARISA:

- :contentReference[oaicite:0]{index=0}

Também foram consultados materiais complementares sobre Árvores de Decisão, Entropia, Ganho de Informação e classificação supervisionada para aprofundamento teórico. :contentReference[oaicite:1]{index=1}

---

# Agradecimentos

Agradeço ao professor **:contentReference[oaicite:2]{index=2}** pela disponibilização do conteúdo didático utilizado como base para o desenvolvimento deste projeto.

O projeto foi expandido a partir do exemplo original, incluindo:

- Dataset próprio para triagem médica;
- Sistema de classificação de urgência (Baixa, Média e Alta);
- Divisão entre treino e teste (*Train/Test Split*);
- Avaliação por Accuracy;
- Matriz de Confusão;
- Visualização gráfica da árvore;
- Interface interativa para entrada de sintomas pelo terminal;
- Estruturação para publicação no GitHub.
