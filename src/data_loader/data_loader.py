# src/data_loader/data_loader.py

import os
import pandas as pd

class DataLoader:
    def __init__(self, dataset_dir='datasets'):
        self.dataset_dir = dataset_dir

    def load_csv(self, filename):
        file_path = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"El archivo {filename} no se encontró en {self.dataset_dir}")
        
        df = pd.read_csv(file_path)
        df.rename(columns={'Unnamed: 0': '#'}, inplace=True)
        return df