# src/tests/test_movie_search.py

import unittest
import pandas as pd
from models.movie_search import MovieSearch
from strategies.certificate_strategy import SpecificCertificateStrategy

class TestMovieSearch(unittest.TestCase):
    
    def setUp(self):
        # Crear un DataFrame simulado
        self.df = pd.DataFrame({
            'Title': ['Movie1', 'Movie2', 'Movie3'],
            'Certificate': ['R', 'PG-13', 'R'],
            'Description': ['Great movie', 'Good movie', 'Okay movie']
        })

    def test_certificate_filtering(self):
        strategy = SpecificCertificateStrategy('R')
        movie_search = MovieSearch(strategy)
        
        filtered_df = strategy.filter_movies(self.df)
        self.assertEqual(len(filtered_df), 2)
        self.assertEqual(filtered_df.iloc[0]['Title'], 'Movie1')

    def test_search_movies(self):
        strategy = SpecificCertificateStrategy('R')
        movie_search = MovieSearch(strategy)
        
        df_embedded = movie_search.compute_embeddings(self.df)
        result = movie_search.search_movies(df_embedded, 'Great')
        
        # Verificar que se devuelven resultados
        self.assertGreater(len(result), 0)

if __name__ == '__main__':
    unittest.main()
