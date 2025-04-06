# 🎯 Predicción de Cáncer de Mama

Este proyecto implementa un modelo de **Random Forest** utilizando el dataset **Breast Cancer Wisconsin Diagnostic** para predecir si un tumor es benigno o maligno a partir de características como el radio, la textura y la simetría del tumor.

---

## 🧠 Modelo Utilizado

- **Tipo de modelo:** Random Forest
- **Motivo:** Alta precisión y buen manejo de datos con relaciones no lineales.

---

## 📊 Dataset

- **Nombre:** Breast Cancer Wisconsin Diagnostic Dataset
- **Fuente:** Incluido en Scikit-Learn (`sklearn.datasets.load_breast_cancer()`)

---

## ⚙️ Variables usadas

- 📏 `radio`: Radio del tumor
- 🧬 `textura`: Textura del tumor
- 🔷 `simetría`: Simetría del tumor

**Variable objetivo (target):**  
- `0`: Tumor Benigno  
- `1`: Tumor Maligno  

---

## 🧪 División del dataset

- **Entrenamiento:** 80%
- **Prueba:** 20%

---

## 🌐 Aplicación Web con Flask

Permite al usuario ingresar los valores del tumor y obtener una predicción visual con el resultado.

---

## ▶️ Cómo ejecutar

1. Crear el entorno virtual e instalar dependencias:
    ```bash
    python -m venv venv
    venv\Scripts\activate  # En Windows
    pip install -r requirements.txt
    ```

2. Ejecutar la aplicación Flask:
    ```bash
    flask run
    ```

---

## 📁 Estructura del Proyecto

