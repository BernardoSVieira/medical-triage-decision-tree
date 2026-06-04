import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Dados de Treinamento no arquivo symptoms_dataset.csv

df = pd.read_csv("data/symptoms_dataset.csv")
print(df.head())

# 2. Pré-processamento (Codificação Categórica)

le_target = LabelEncoder()
df["Urgencia"] = le_target.fit_transform(df["Urgencia"])

X = df[["Febre", "Tosse", "DorPeito", "FaltaAr", "DorCabeca", "Nausea"]]
y = df["Urgencia"]

# 3. Treinamento do Modelo (Critério ID3: Entropia)
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X, y)

# --- SEÇÃO DE TESTE (SAÍDAS) ---
print("--- Simulador de Decisão ---")
# Exemplo: Paciente com Febre=Sim, Tosse=Não e DorPeito=Não
nova_entrada = pd.DataFrame(
    [[1, 0, 0, 0, 0, 0]],
    columns=["Febre", "Tosse", "DorPeito", "FaltaAr", "DorCabeca", "Nausea"]
)
predicao = clf.predict(nova_entrada)
resultado = le_target.inverse_transform(predicao)

print(f"Resultado para Febre: Sim, Tosse: Nao, Dor no Peito: Nao: {resultado[0]}")

# 4. Visualização
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=le_target.classes_,
    filled=True,
)
print(le_target.classes_)

plt.show()