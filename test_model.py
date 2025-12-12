import os
import joblib
import numpy as np

# ====== 1. Charger le modèle entraîné (CHEMIN RELATIF) ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_infecte.pkl")

print("🔍 Chargement du modèle...")
model = joblib.load(MODEL_PATH)
print("✅ Modèle chargé avec succès !\n")

# ====== 2. Exemple de données pour tester ======
temperature = 38.5
tension = 120
toux = 1

sample = np.array([[temperature, tension, toux]])

# ====== 3. Faire la prédiction ======
prediction = model.predict(sample)[0]

# ====== 4. Interprétation du résultat ======
label = "INFECTÉ" if prediction == 1 else "NON INFECTÉ"

print(f"🔬 Données testées : température={temperature}, tension={tension}, toux={toux}")
print(f"🧪 Résultat prédiction : {prediction} → {label}")
