from core.dataset import PatientDataset
from core.logistic_regression import LogisticRegressionModel

class Trainer:
    def __init__(self, data_path):
        self.dataset = PatientDataset(data_path)
        self.model = LogisticRegressionModel()

    def train_model(self):
        # 1. charger les données
        self.dataset.load()
        # 2. séparer X (features) et y (labels)
        X, y = self.dataset.get_features_and_labels()
        # 3. entraîner le modèle
        self.model.train(X, y)
        # 4. retourner le modèle entraîné
        return self.model


if __name__ == "__main__":
    trainer = Trainer("data/patients.csv")
    trained_model = trainer.train_model()
    print("✅ Modèle entraîné avec succès !")
