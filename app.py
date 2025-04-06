from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

# Cargar dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# Variables independientes y dependiente
X = df[["mean radius", "mean texture", "mean symmetry"]]
y = df["target"]

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de clasificación Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
matriz_confusion = confusion_matrix(y_test, y_pred)

# Función para convertir la matriz a imagen
def generar_matriz_confusion(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.set_title("Matriz de Confusión")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    probabilidad = None
    imagen_cm = None

    if request.method == "POST":
        radio = float(request.form["radio"])
        textura = float(request.form["textura"])
        simetria = float(request.form["simetria"])

        entrada = np.array([[radio, textura, simetria]])
        prediccion = model.predict(entrada)[0]
        proba = model.predict_proba(entrada)[0][int(prediccion)]

        resultado = "maligno" if prediccion == 1 else "benigno"
        probabilidad = round(proba * 100, 2)
        imagen_cm = generar_matriz_confusion(matriz_confusion)

    return render_template(
        "index.html",
        resultado=resultado,
        probabilidad=probabilidad,
        accuracy=round(accuracy * 100, 2),
        precision=round(precision * 100, 2),
        recall=round(recall * 100, 2),
        imagen_cm=imagen_cm
    )

if __name__ == "__main__":
    app.run(debug=True)
