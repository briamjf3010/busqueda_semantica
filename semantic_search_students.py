## Instalando dependencias necesarias
#%pip install -U sentence-transformers pandas

import pandas as pd
## Entendiendo el dataset
# https://www.kaggle.com/datasets/omarhanyy/imdb-top-1000?resource=download
# TODO: Cargar el archivo del dataset con  
# Cargar el archivo del dataset con pandas
df = pd.read_csv('C:\\Users\\DELL\\OneDrive - UPB\\Desktop\\DESARROLLO_PROYECTOS_IA\\IMDB top 1000.csv')
#df = pd.read_csv('src/app/IMDB top 1000.csv')

# TODO: mostrar los primeros 5 registros de dataframe
df.head()
df.rename(columns={'Unnamed: 0': '#'}, inplace=True)
df.info()
import matplotlib.pyplot as plt
import seaborn as sns

# Convertir 'Duration' en formato numérico quitando 'min'
df['Duration'] = df['Duration'].str.replace(' min', '').astype(int)

# Graficar la distribución de la duración
plt.figure(figsize=(10,6))
sns.histplot(df['Duration'], bins=20, kde=True)
plt.title('Distribución de la duración de las películas')
plt.xlabel('Duración (minutos)')
plt.ylabel('Frecuencia')
plt.show()

# Graficar la relación entre Rate y Duration
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='Duration', y='Rate')
plt.title('Relación entre duración y calificación')
plt.xlabel('Duración (minutos)')
plt.ylabel('Calificación')
plt.show()


# Función para convertir el texto del precio a un número
def extract_price(info):
    if "Gross" in info:
        try:
            # Extraemos el valor bruto después de 'Gross:'
            price_str = info.split("Gross:")[1].strip()
            
            # Convertimos el valor a millones o miles según el sufijo 'M' o 'K'
            if price_str[-1] == 'M':
                return float(price_str[1:-1]) * 1e6  # Elimina el símbolo '$' y 'M', convierte a número
            elif price_str[-1] == 'K':
                return float(price_str[1:-1]) * 1e3  # Elimina el símbolo '$' y 'K', convierte a número
        except:
            return None
    return None

# Aplicar la función a la columna 'Info' para obtener el precio bruto
df['Gross_Price'] = df['Info'].apply(extract_price)

import plotly.graph_objects as go

# Limpiar el dataset eliminando filas con valores nulos en 'Rate', 'Duration', y 'Gross_Price'
df_clean = df.dropna(subset=['Rate', 'Duration', 'Gross_Price'])

# Crear la figura en 3D con Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=df_clean['Rate'],          # Calificación de la película
    y=df_clean['Duration'],      # Duración de la película
    z=df_clean['Gross_Price'],   # Ingresos brutos
    mode='markers',              # Modo de los puntos (marcadores)
    marker=dict(
        size=8,                  # Tamaño de los puntos
        color=df_clean['Gross_Price'],  # Color de los puntos en función de los ingresos
        colorscale='Viridis',     # Escala de color 'viridis'
        opacity=0.8,              # Opacidad de los puntos
        colorbar=dict(title='Ingresos Brutos')  # Etiqueta de la barra de color
    )
)])

# Añadir etiquetas de los ejes
fig.update_layout(
    scene=dict(
        xaxis_title='Calificación (Rate)',
        yaxis_title='Duración (minutos)',
        zaxis_title='Ingresos Brutos (Gross Price)',
    ),
    title='Relación entre Calificación, Duración y Valor Bruto de las Películas',
    margin=dict(l=0, r=0, b=0, t=40)  # Márgenes de la gráfica
)

# Mostrar la gráfica
fig.show()



# Limpiar el dataset eliminando filas con valores nulos en 'Rate', 'Duration', y 'Gross_Price'
df_clean = df.dropna(subset=['Metascore', 'Duration', 'Gross_Price'])

# Crear la figura en 3D con Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=df_clean['Metascore'],          # Calificación de la película
    y=df_clean['Duration'],      # Duración de la película
    z=df_clean['Gross_Price'],   # Ingresos brutos
    mode='markers',              # Modo de los puntos (marcadores)
    marker=dict(
        size=8,                  # Tamaño de los puntos
        color=df_clean['Metascore'],  # Color de los puntos en función de los ingresos
        colorscale='Viridis',     # Escala de color 'viridis'
        opacity=0.8,              # Opacidad de los puntos
        colorbar=dict(title='Metascore')  # Etiqueta de la barra de color
    )
)])

# Añadir etiquetas de los ejes
fig.update_layout(
    scene=dict(
        xaxis_title='Metascore',
        yaxis_title='Duración (minutos)',
        zaxis_title='Ingresos Brutos (Gross Price)',
    ),
    title='Relación entre Calificación, Duración y Valor Bruto de las Películas',
    margin=dict(l=0, r=0, b=0, t=40)  # Márgenes de la gráfica
)

# Mostrar la gráfica
fig.show()

# Separar los géneros que están en una lista separada por comas
df['Genre'] = df['Genre'].str.split(', ')

# Explorar las películas con mejor calificación por género
top_movies_by_genre = df.explode('Genre').groupby('Genre')['Rate'].mean().sort_values(ascending=False)

# Graficar
plt.figure(figsize=(10,6))
sns.barplot(x=top_movies_by_genre.index, y=top_movies_by_genre.values)
plt.xticks(rotation=90)
plt.title('Calificación media por género')
plt.xlabel('Género')
plt.ylabel('Calificación media')
plt.show()

# Graficar la distribución del Metascore
plt.figure(figsize=(10,6))
sns.histplot(df['Metascore'], bins=20, kde=True)
plt.title('Distribución de Metascore')
plt.xlabel('Metascore')
plt.ylabel('Frecuencia')
plt.show()

# Directores más frecuentes
df['Director'] = df['Cast'].str.extract(r'Director:\s*(.+?)\s*\|')
top_directors = df['Director'].value_counts().head(10)

# Graficar directores más frecuentes
plt.figure(figsize=(10,6))
sns.barplot(x=top_directors.values, y=top_directors.index)
plt.title('Top 10 directores más frecuentes')
plt.xlabel('Número de películas')
plt.ylabel('Director')
plt.show()

# Extraer los directores de la columna 'Cast'
df['Director'] = df['Cast'].str.extract(r'Director:\s*(.+?)\s*\|')

# Calcular el promedio de calificación por director
avg_rate_by_director = df.groupby('Director')['Rate'].mean().sort_values(ascending=False)

# Mostrar los primeros 10 directores con mejor promedio de calificación
top_10_directors = avg_rate_by_director.head(10)
print(top_10_directors)


# Graficar los 10 directores con mayor promedio de calificación
plt.figure(figsize=(10,6))
sns.barplot(x=top_10_directors.values, y=top_10_directors.index, hue=top_10_directors.index, palette='viridis', dodge=False, legend=False)
plt.title('Top 10 Directores por Promedio de Calificación')
plt.xlabel('Promedio de Calificación')
plt.ylabel('Director')
plt.show()

from sentence_transformers import SentenceTransformer, util

# Cargar el dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# Calcular embeddings para la columna Description
def compute_embeddings(df):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(df['Description'], batch_size=64, show_progress_bar=True)
    df['embeddings'] = embeddings.tolist()
    return df, model

# Calcular la similitud entre los embeddings
def search_movies(df, model, query):
    # Convertir la consulta a un embedding
    query_embedding = model.encode([query])[0]
    
    # Definir función para calcular la similitud coseno
    def compute_similarity(example):
        embedding = example['embeddings']
        similarity = util.cos_sim(embedding, query_embedding).item()
        return similarity

    # Aplicar la función de similitud a todo el dataframe
    df['similarity'] = df.apply(compute_similarity, axis=1)

    # Ordenar los resultados por similitud
    df_sorted = df.sort_values(by='similarity', ascending=False)

    # Retornar los títulos de las películas más similares
    return df_sorted[['Title', 'similarity']].head()