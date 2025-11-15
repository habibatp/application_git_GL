from pipeline.trainer import Trainer
from core.dataset import PatientDataset

class ClinicalPredictor:
    def __init__(self, model):
        """
        model : n'importe quel modèle IA qui a .predict_proba()
                et renvoie une proba d'infection.
        """
        self.model = model

    def diagnostic(self, patient_data):
        """
        patient_data : [temperature, tension, toux]
        retourne "infecté" si proba > 0.5 sinon "sain"
        """
        # sklearn attend une liste de patients, pas un seul
        proba_infected = self.model.predict_proba([patient_data])[0]

        # seuil 0.5
        return "infecté" if proba_infected > 0.5 else "sain"


if __name__ == "__main__":
    # 1. Entraîner le modèle une fois
    trainer = Trainer("C:/Users/pc/PycharmProjects/exercice_dapplication_GL/data/patients.csv")
    trained_model = trainer.train_model()

    # 2. Charger le dataset brut
    dataset = PatientDataset("C:/Users/pc/PycharmProjects/exercice_dapplication_GL/data/patients.csv")
    dataset.load()

    # 3. Récupérer les features d'un patient (ex: id = 2)
    features_patient_2 = dataset.get_patient_features(2)

    # 4. Créer notre outil clinique
    predictor = ClinicalPredictor(trained_model)

    # 5. Afficher le résultat
    print("Caractéristiques patient 2 :", features_patient_2)
    print("Diagnostic :", predictor.diagnostic(features_patient_2))
