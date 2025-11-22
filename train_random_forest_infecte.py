import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# ========= 1. Charger le dataset =========
# ✅ Chemin corrigé (utilise des /)
CSV_PATH = "C:/Users/User/Desktop/AMIZMIZ_HABIBATOU_ALLAH_POO_EXERCICE/exercice_dapplication_GL/data/patients_infectes.csv"
# Autre option possible :
# CSV_PATH = r"C:\Users\User\Desktop\AMIZMIZ_HABIBATOU_ALLAH_POO_EXERCICE\exercice_dapplication_GL\data\patients_infectes.csv"

df = pd.read_csv(CSV_PATH)

print("Aperçu du dataset :")
print(df.head())

# ========= 2. Nettoyage / sélection des colonnes =========
# On enlève la colonne id (pas utile pour la prédiction)
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Vérifier qu'on a bien les colonnes attendues
print("\nColonnes présentes :", df.columns.tolist())

# ========= 3. Séparer X (features) et y (label) =========
# Features : temperature, tension, toux
X = df[["temperature", "tension", "toux"]]

# Cible : infecte
y = df["infecte"]

# Si 'infecte' est sous forme texte (ex: "oui"/"non"), on encode en 0/1
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)
    # Sauvegarder l'encodeur aussi si tu veux le réutiliser
    joblib.dump(le, "label_encoder_infecte.pkl")
    print("\nEncodage de la variable 'infecte' :", dict(zip(le.classes_, le.transform(le.classes_))))

# ========= 4. Train / Test Split =========
# ========= 4. Train / Test Split =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTaille train : {X_train.shape[0]}  |  Taille test : {X_test.shape[0]}")



# ========= 5. Définir et entraîner le Random Forest =========
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# ========= 6. Évaluation du modèle =========
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy sur le test : {acc:.2f}")

print("\n📊 Matrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\n📄 Classification report :")
print(classification_report(y_test, y_pred))

# ========= 7. Sauvegarder le modèle en .pkl =========
MODEL_PATH = "random_forest_infecte.pkl"
joblib.dump(model, MODEL_PATH)

print(f"\n💾 Modèle sauvegardé dans : {MODEL_PATH}")

# ========= 8. Exemple de prédiction manuelle =========
# Exemple : temperature=38.5, tension=12.5, toux=1
sample = np.array([[38.5, 12.5, 1]])  # shape (1, 3) comme X

pred_sample = model.predict(sample)[0]
print("\n🔍 Prédiction pour [38.5, 12.5, toux=1] :", pred_sample)
