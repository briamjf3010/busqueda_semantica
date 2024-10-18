# src/strategies/base_strategy.py

from abc import ABC, abstractmethod

class CertificateStrategy(ABC):
    @abstractmethod
    def filter_movies(self, df):
        pass
