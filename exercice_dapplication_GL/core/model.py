from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y):
        """Entraîner le modèle sur les features X et les labels y."""
        pass

    @abstractmethod
    def predict(self, X):
        """Prédire la classe (0 ou 1). Retourne une liste ou un array."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Donner la probabilité d'être infecté.
        Retourne une liste/array de probabilités entre 0 et 1.
        """
        pass
