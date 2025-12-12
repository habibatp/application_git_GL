import os
import joblib
import numpy as np

def test_model_prediction():
    # Chemin relatif vers le modèle
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "random_forest_infecte.pkl")

    # Charger le modèle
    model = joblib.load(model_path)

    # Données de test
    sample = np.array([[38.5, 120, 1]])

    # Prédiction
    prediction = model.predict(sample)

    # Assertions obligatoires pour pytest
    assert prediction is not None
    assert prediction[0] in [0, 1]
