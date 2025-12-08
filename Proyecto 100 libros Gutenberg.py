#!/usr/bin/env python
# coding: utf-8

# ### Usa dataframes y soporta bien los libros

# In[1]:


# ================================
# Importar librerías
# ================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, lower, explode, regexp_replace, col, concat_ws
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.linalg import SparseVector
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

# ================================
# Crear SparkSession
# ================================
spark = SparkSession.builder.appName("RecomendacionLibros").master("local[*]").getOrCreate()
sc = spark.sparkContext
import os

# Ruta de los archivos
directorio = "Libros_clean" 

if os.path.exists(directorio):
    for filename in os.listdir(directorio):
        # Si el archivo tiene dos puntos, lo renombramos
        if ":" in filename:
            nuevo_nombre = filename.replace(":", "") # Simplemente borramos los dos puntos
            src = os.path.join(directorio, filename)
            dst = os.path.join(directorio, nuevo_nombre)
            os.rename(src, dst)
# ================================
# Cargar archivos de texto
# ================================
directorio = "Libros_clean"
rdd = sc.wholeTextFiles(f"{directorio}/*.txt")
df = rdd.map(lambda x: (os.path.basename(x[0]), x[1])).toDF(["Titulo", "Contenido"])

# ================================
# Limpiar texto (quitar caracteres especiales)
# ================================
def clean(df, col):
    return df.withColumn(col, regexp_replace(col, "[^a-zA-Z0-9\\s]", ""))

df_cleaned = clean(df, "Contenido")

# ================================
# Convertir a array de palabras
# ================================
df_array = df_cleaned.withColumn("Contenido_Array", split(lower(col("Contenido")), "\\s+"))

# ================================
# Quitar stopwords
# ================================
remover = StopWordsRemover(inputCol="Contenido_Array", outputCol="Contenido_Limpio")
df_no_stop = remover.transform(df_array)

# ================================
# CountVectorizer + IDF
# ================================
cv = CountVectorizer(inputCol="Contenido_Limpio", outputCol="raw_features", vocabSize=10000)
cv_model = cv.fit(df_no_stop)
df_vectorized = cv_model.transform(df_no_stop)

idf = IDF(inputCol="raw_features", outputCol="tfidf_features")
idf_model = idf.fit(df_vectorized)
df_tfidf = idf_model.transform(df_vectorized)

# ================================
# Matriz de similitud libro x libro
# ================================
# Materializar TODO de una vez
data = df_tfidf.select("Titulo", "tfidf_features", "Contenido_Limpio").collect()

titulos = [row["Titulo"] for row in data]
vectores = []
palabras_por_libro = {}  # ← Guardar las palabras aquí

for row in data:
    titulo = row["Titulo"]
    vec = row["tfidf_features"]
    palabras_por_libro[titulo] = row["Contenido_Limpio"]  # ← Guardar palabras
    vectores.append(vec.toArray() if isinstance(vec, SparseVector) else np.array(vec))

X = np.vstack(vectores)
sim_matrix = cosine_similarity(X)
df_sim = pd.DataFrame(sim_matrix, index=titulos, columns=titulos)
# ================================
# Función de recomendación
# ================================
def recomendar_libros(df_sim, nombre_libro, cantidad):
    if nombre_libro not in df_sim.columns:
        raise ValueError(f"El libro {nombre_libro} no está en la matriz de similitud.")
    similitudes = df_sim.loc[nombre_libro]
    recomendados = similitudes.drop(nombre_libro).sort_values(ascending=False)
    return recomendados.head(cantidad)

# ================================
# Función para top palabras por libro
# ================================
def top_palabras(title, n):
    if title not in palabras_por_libro:
        raise ValueError(f"El libro {title} no se encontró en los datos.")

    # Usar los datos ya materializados
    words = palabras_por_libro[title]

    # Encontrar el índice del libro
    idx = titulos.index(title)
    tfidf_vec = vectores[idx]  # Ya está en formato array

    # Si es SparseVector, convertir
    if isinstance(tfidf_vec, np.ndarray):
        # Necesitamos los índices y valores
        # Obtener el vector original antes de convertir
        tfidf_row_original = [row for row in data if row["Titulo"] == title][0]["tfidf_features"]
        if isinstance(tfidf_row_original, SparseVector):
            word_scores = [(words[idx], value) for idx, value in zip(tfidf_row_original.indices, tfidf_row_original.values)]
        else:
            # Si es denso, crear índices manualmente
            word_scores = [(words[i], tfidf_vec[i]) for i in range(len(tfidf_vec)) if i < len(words) and tfidf_vec[i] > 0]

    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [w for w, score in word_scores[:n]]


# In[2]:


# ================================
# Vocabulario usando solo los datos ya procesados
# ================================

# Vocabulario total: todas las palabras de todos los libros (incluye repeticiones)
vocabulario_total = [w for lst in palabras_por_libro.values() for w in lst]

# Vocabulario limpio único: todas las palabras de todos los libros, sin duplicados
vocabulario_unico_limpio = set(vocabulario_total)


# In[ ]:


# ================================
# Celda Final: Recomendaciones y Vocabulario
# ================================

# Función 1️⃣: libros más parecidos
def recomendar_libros_usuario():
    libro = input("Ingresa el nombre del libro: ")
    n = int(input("Cantidad de libros a recomendar: "))

    if libro not in df_sim.columns:
        print(f"Libro '{libro}' no encontrado.")
        return

    similitudes = df_sim.loc[libro].drop(libro)
    top_n = similitudes.sort_values(ascending=False).head(n)
    print(f"\nLos {n} libros más parecidos a '{libro}' son:")
    print(top_n)

# Función 2️⃣: top palabras representativas
def top_palabras_usuario():
    libro = input("Ingresa el nombre del libro: ")
    n = int(input("Cantidad de palabras representativas: "))

    if libro not in palabras_por_libro:
        print(f"Libro '{libro}' no encontrado.")
        return

    words = palabras_por_libro[libro]
    idx = titulos.index(libro)
    tfidf_vec = vectores[idx]

    if isinstance(tfidf_vec, np.ndarray):
        tfidf_row_original = [row for row in data if row["Titulo"] == libro][0]["tfidf_features"]
        if hasattr(tfidf_row_original, "indices"):
            word_scores = [(words[i], v) for i, v in zip(tfidf_row_original.indices, tfidf_row_original.values)]
        else:
            word_scores = [(words[i], tfidf_vec[i]) for i in range(len(tfidf_vec)) if i < len(words) and tfidf_vec[i] > 0]

    word_scores.sort(key=lambda x: x[1], reverse=True)
    top_words = [w for w, _ in word_scores[:n]]
    print(f"\nLas {n} palabras más representativas de '{libro}' son:")
    print(top_words)

# Función 3️⃣: mostrar vocabulario
def mostrar_vocabulario():
    opcion = input("Mostrar vocabulario completo o solo limpio? (c/l): ").lower()
    if opcion == "c":
        print(f"\nVocabulario completo ({len(vocabulario_total)} palabras):")
        print(vocabulario_total[:100])
    elif opcion == "l":
        print(f"\nVocabulario limpio ({len(vocabulario_unico_limpio)} palabras):")
        print(list(vocabulario_unico_limpio)[:100])  # <--- usa la variable que ya estaba limpia
    else:
        print("Opción no válida. Use 'c' para completo o 'l' para limpio.")
# Función 4️⃣: mostrar matriz de similitud limitada
def mostrar_matriz_similitud():
    print("\nMatriz de similitud libro x libro (limitada a los primeros 10 libros):")
    df_sim_limited = df_sim.iloc[:10, :10]  # solo primeros 10 libros
    print(df_sim_limited)

# ================================
# Menú interactivo
# ================================
while True:
    print("\nOpciones:")
    print("1 - Recomendar libros similares")
    print("2 - Obtener top palabras de un libro")
    print("3 - Mostrar vocabulario")
    print("4 - Mostrar matriz de similitud (primeros 10 libros)")
    print("0 - Salir")

    opcion = input("Seleccione una opción: ")
    if opcion == "1":
        recomendar_libros_usuario()
    elif opcion == "2":
        top_palabras_usuario()
    elif opcion == "3":
        mostrar_vocabulario()
    elif opcion == "4":
        mostrar_matriz_similitud()
    elif opcion == "0":
        break
    else:
        print("Opción no válida.")


# In[ ]:




