## Instalando dependencias necesarias
#%pip install -U sentence-transformers pandas

import pandas as pd

# Cargar el archivo del dataset con pandas
#df = pd.read_csv('IMDBtop1000.csv')
#df = pd.read_csv('src/app/IMDB top 1000.csv')


from sentence_transformers import SentenceTransformer, util

# Cargar el dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.rename(columns={'Unnamed: 0': '#'}, inplace=True)
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