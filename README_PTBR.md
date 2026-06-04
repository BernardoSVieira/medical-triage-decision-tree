
---

# README_PTBR.md

```md
# 🌳 Árvore de Decisão para Triagem Médica

Projeto de Machine Learning que simula um sistema de triagem médica utilizando uma Árvore de Decisão.

O modelo recebe sintomas de um paciente e prevê o nível de urgência:

- 🟢 Baixa Urgência
- 🟡 Média Urgência
- 🔴 Alta Urgência

O objetivo do projeto é estudar Árvores de Decisão, Ganho de Informação (Entropia), avaliação de modelos e sistemas de classificação utilizando Scikit-Learn.

---

## Funcionalidades

- Geração de dataset sintético
- Treinamento de Árvore de Decisão utilizando Entropia
- Avaliação do modelo através de:
  - Acurácia
  - Matriz de Confusão
  - Relatório de Classificação
- Visualização da árvore gerada
- Sistema interativo de triagem no terminal

---

## Sintomas Utilizados

| Sintoma | Tipo |
|----------|----------|
| Febre | Binário |
| Tosse | Binário |
| Dor no Peito | Binário |
| Falta de Ar | Binário |
| Dor de Cabeça | Binário |
| Náusea | Binário |

---

## Níveis de Urgência

| Valor | Classificação |
|---------|---------|
| 0 | Baixa |
| 1 | Média |
| 2 | Alta |

---

## Tecnologias

- Python
- Scikit-Learn
- Pandas
- NumPy
- Matplotlib
