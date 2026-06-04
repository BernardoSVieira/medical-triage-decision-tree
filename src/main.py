import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 1. Dados de Treinamento
data = {
    "Febre": [
        "Sim", "Sim", "Nao", "Nao",
        "Sim", "Nao", "Sim", "Nao"
    ],

    "Tosse": [
        "Sim", "Nao", "Sim", "Nao",
        "Sim", "Nao", "Nao", "Sim"
    ],

    "DorPeito": [
        "Nao", "Nao", "Nao", "Nao",
        "Sim", "Nao", "Sim", "Sim"
    ],

    "Urgencia": [
        "Media",
        "Baixa",
        "Baixa",
        "Baixa",
        "Alta",
        "Baixa",
        "Alta",
        "Media"
    ]
}
df = pd.DataFrame(data)

# 2. Pré-processamento (Codificação Categórica)
le_febre = LabelEncoder()
le_tosse = LabelEncoder()
le_dor_peito = LabelEncoder()
le_target = LabelEncoder()

df["febre"] = le_febre.fit_transform(df["Febre"])
df["tosse"] = le_tosse.fit_transform(df["Tosse"])
df["dor_peito"] = le_dor_peito.fit_transform(df["DorPeito"])
df["Urgencia"] = le_target.fit_transform(df["Urgencia"])

X = df[["febre", "tosse", "dor_peito"]]
y = df["Urgencia"]

# 3. Treinamento do Modelo (Critério ID3: Entropia)
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X, y)

# --- SEÇÃO DE TESTE (SAÍDAS) ---
print("--- Simulador de Decisão ---")
# Exemplo: Paciente com Febre=Sim, Tosse=Não e DorPeito=Não
nova_entrada = [[1, 0, 0]]
predicao = clf.predict(nova_entrada)
resultado = le_target.inverse_transform(predicao)

print(f"Resultado para Febre: Sim, Tosse: Nao, Dor no Peito: Nao: {resultado[0]}")

# 4. Visualização
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["Baixa", "Media", "Alta"],
    filled=True,
)
plt.show()