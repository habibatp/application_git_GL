import csv

class PatientDataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = []

    def load(self):
        """Charge les données du CSV en mémoire."""
        with open(self.filepath, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.data = [row for row in reader]
        return self.data

    def get_features_and_labels(self):
        """Retourne X (features) et y (étiquette infecté ou pas)."""
        X, y = [], []
        for row in self.data:
            X.append([
                float(row["temperature"]),
                float(row["tension"]),
                int(row["toux"])
            ])
            y.append(int(row["infecte"]))
        return X, y

    def get_patient_features(self, patient_id):
        """Retourne les features [temp, tension, toux] pour 1 patient."""
        for row in self.data:
            if int(row["id"]) == patient_id:
                return [
                    float(row["temperature"]),
                    float(row["tension"]),
                    int(row["toux"])
                ]
        return None
