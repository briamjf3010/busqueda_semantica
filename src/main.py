# main.py

from models.movie_search import MovieSearch
from strategies.certificate_strategy import SpecificCertificateStrategy

def main():
    # Pedir al usuario que seleccione un "Certificate"
    print("Elige un 'Certificate' para filtrar las películas (por ejemplo: R, PG-13, G, Not Rated):")
    user_certificate = input()

    # Crear una estrategia basada en el "Certificate"
    strategy = SpecificCertificateStrategy(user_certificate)

    # Crear el objeto MovieSearch con la estrategia seleccionada
    movie_search = MovieSearch(strategy)

    # Cargar el dataset
    df = movie_search.load_data('IMDBtop1000.csv')  

    # Computar embeddings para el DataFrame cargado
    df_embedded = movie_search.compute_embeddings(df)

    # Búsqueda de películas
    query = input("Escribe una consulta para buscar películas (por ejemplo: 'great movie'):")
    results = movie_search.search_movies(df_embedded, query)

    # Mostrar resultados
    print("Películas recomendadas:")
    print(results)

if __name__ == '__main__':
    main()
