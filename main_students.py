import pandas as pd
import semantic_search_students as ss  # Importar las funciones del otro archivo

def main(query):
    # Cargar el dataset
    df = ss.load_data('.\src\IMDB top 1000.csv')
    
    # Calcular embeddings (solo debe hacerse una vez)
    df, model = ss.compute_embeddings(df)
    
    # Realizar la búsqueda con el término proporcionado
    result = ss.search_movies(df, model, query)
    
    # Mostrar los resultados
    print(result)

if __name__ == '__main__':
    query = input('Ingresa el término de búsqueda: ')
    main(query)
