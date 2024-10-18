# src/strategies/certificate_strategy.py

import pandas as pd
from strategies.base_strategy import CertificateStrategy

class SpecificCertificateStrategy(CertificateStrategy):
    def __init__(self, certificate):
        self.certificate = certificate

    def filter_movies(self, df: pd.DataFrame):
        # Filtrar por el certificado seleccionado
        filtered_df = df[df['Certificate'] == self.certificate]
        return filtered_df
