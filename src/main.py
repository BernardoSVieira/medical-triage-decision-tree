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
# Divisão dos dados em treino e teste 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

print(f"Accuracy: {accuracy:.2f}")

# --- SEÇÃO DE TESTE (SAÍDAS) ---

print("\n====== Triagem Médica - Árvore de Decisão (ID3) ======\n")

sintomas = [
    "Febre",
    "Tosse",
    "DorPeito",
    "FaltaAr",
    "DorCabeca",
    "Nausea"
]

respostas = []

contador = 0

while contador < len(sintomas):

    input_sintoma = input(
        f'''Paciente possui {sintomas[contador]}?\n 
        [S] Sim
        [N] Não \n'''
    )

    if input_sintoma.lower() not in ["s", "n"]:
        print("Entrada inválida. Digite apenas S ou N.\n")
        continue

    if input_sintoma.lower() == "s":
        respostas.append(1)
    elif input_sintoma.lower() == "n":
        respostas.append(0)
    else:
        print("Entrada inválida. Digite apenas S ou N.\n")
        continue


    contador += 1


nova_entrada = pd.DataFrame(
    [respostas],
    columns=sintomas
)

print("\nRespostas capturadas:")
print(respostas)

print("\nEntrada enviada ao modelo:")
print(nova_entrada)


predicao = clf.predict(nova_entrada)

resultado = le_target.inverse_transform(predicao)

print("\n==============================")
print(" ===== RESULTADO DA TRIAGEM =====")
print("==============================")
print(f"Urgência prevista: {resultado[0]}")


# 4. Visualização
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=le_target.classes_,
    filled=True,
)
print(le_target.classes_)

plt.show()


from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    clf,
    X_test,
    y_test
)

plt.show()