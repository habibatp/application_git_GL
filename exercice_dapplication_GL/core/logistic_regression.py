from sklearn.linear_model import LogisticRegression
from core.model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        # max_iter ↑ pour être sûr que l'entraînement converge
        self.model = LogisticRegression(max_iter=200)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        # renvoie 0 ou 1
        return self.model.predict(X)

    def predict_proba(self, X):
        # renvoie la proba d'être en classe 1 (= infecté)
        # sklearn renvoie [[p0, p1], [p0, p1], ...]
        # on prend juste la colonne p1
        proba_all = self.model.predict_proba(X)
        return proba_all[:, 1]
