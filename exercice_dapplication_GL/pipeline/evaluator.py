from core.dataset import PatientDataset
from utils.metrics import compute_metrics
from core.loss import LossFunction

class Evaluator:
    def __init__(self, model, data_path):
        self.model = model
        self.dataset = PatientDataset(data_path)
        self.loss_fn = LossFunction()

    def evaluate(self):
        # Charger les données
        self.dataset.load()
        X, y_true = self.dataset.get_features_and_labels()

        # Prédictions classe (0/1)
        y_pred = self.model.predict(X)

        # Probabilités d'être infecté
        y_pred_proba = self.model.predict_proba(X)

        # Calcul des métriques
        scores = compute_metrics(y_true, y_pred)

        # Calcul de la loss
        scores["log_loss"] = self.loss_fn.compute(y_true, y_pred_proba)

        return scores


if __name__ == "__main__":
    from pipeline.trainer import Trainer
    trainer = Trainer("data/patients.csv")
    model = trainer.train_model()

    evaluator = Evaluator(model, "data/patients.csv")
    report = evaluator.evaluate()
    print("Rapport d'évaluation :", report)
