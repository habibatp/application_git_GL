class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def optimize(self, model):
        # Dans un vrai projet deep learning, on ferait un pas de gradient ici.
        # Ici on affiche juste pour montrer le rôle.
        print(f"Optimisation terminée (learning_rate = {self.learning_rate})")
