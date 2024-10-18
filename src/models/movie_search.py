# src/models/movie_search.py

from sentence_transformers import SentenceTransformer
from strategies.base_strategy import CertificateStrategy
from data_loader.data_loader import DataLoader

class MovieSearch:
    def __init__(self, strategy: CertificateStrategy):
        self.strategy = strategy
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def load_data(self, filename: str):
        # Utilizar el DataLoader para cargar los datos
        loader = DataLoader()
        df = loader.load_csv(filename)
        return df

    def compute_embeddings(self, df):
        embeddings = self.model.encode(df['Description'], batch_size=64, show_progress_bar=True)
        df['embeddings'] = embeddings.tolist()
        return df

    def search_movies(self, df, query):
        # Filtrar películas usando la estrategia
        df_filtered = self.strategy.filter_movies(df)

        # Convertir la consulta a un embedding
        query_embedding = self.model.encode([query])[0]

        # Calcular la similitud coseno
        df_filtered['similarity'] = df_filtered.apply(
            lambda row: util.cos_sim(row['embeddings'], query_embedding).item(), axis=1
        )

        # Ordenar las películas por similitud
        df_sorted = df_filtered.sort_values(by='similarity', ascending=False)
        return df_sorted[['Title', 'similarity']].head()