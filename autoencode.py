import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class FullScratchAutoencoder:
    def __init__(self, input_dim, encoding_dim, learning_rate=0.01, epochs=100):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialisation des poids (Xavier/Glorot)
        self.W_encoder = np.random.randn(input_dim, encoding_dim) * np.sqrt(2.0 / input_dim)
        self.b_encoder = np.zeros((1, encoding_dim))

        self.W_decoder = np.random.randn(encoding_dim, input_dim) * np.sqrt(2.0 / encoding_dim)
        self.b_decoder = np.zeros((1, input_dim))

        # Historique d'entraînement
        self.train_loss_history = []
        self.val_loss_history = []

    def sigmoid(self, x):
        """Fonction d'activation sigmoid avec clipping pour éviter l'overflow"""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Dérivée de la fonction sigmoid"""
        return x * (1 - x)

    def encode(self, X):
        """Encodage: X -> espace latent"""
        z = np.dot(X, self.W_encoder) + self.b_encoder
        return self.sigmoid(z)

    def decode(self, encoded):
        """Décodage: espace latent -> X"""
        z = np.dot(encoded, self.W_decoder) + self.b_decoder
        return self.sigmoid(z)

    def forward(self, X):
        """Passe avant complète"""
        encoded = self.encode(X)
        decoded = self.decode(encoded)
        return encoded, decoded

    def compute_loss(self, X, X_reconstructed):
        """Calcul de la perte (MSE)"""
        return np.mean((X - X_reconstructed) ** 2)

    def fit(self, X_train, X_val=None, verbose=True):
        """
        Entraînement de l'autoencoder

        Args:
            X_train: Données d'entraînement
            X_val: Données de validation (optionnel)
            verbose: Afficher le progrès
        """
        if verbose:
            print(f"🔄 Entraînement de l'autoencoder ({self.epochs} epochs)...")
            print(f"   • Dimension d'entrée: {self.input_dim}")
            print(f"   • Dimension d'encodage: {self.encoding_dim}")
            print(f"   • Taux d'apprentissage: {self.learning_rate}")

        for epoch in range(self.epochs):
            # Passe avant
            encoded, decoded = self.forward(X_train)

            # Calcul de la perte
            train_loss = self.compute_loss(X_train, decoded)
            self.train_loss_history.append(train_loss)

            # Passe arrière (backpropagation)
            m = X_train.shape[0]

            # Erreur de sortie
            output_error = decoded - X_train

            # Gradients pour le décodeur
            dW_decoder = np.dot(encoded.T, output_error) / m
            db_decoder = np.sum(output_error, axis=0, keepdims=True) / m

            # Erreur propagée vers l'encodeur
            hidden_error = np.dot(output_error, self.W_decoder.T) * self.sigmoid_derivative(encoded)

            # Gradients pour l'encodeur
            dW_encoder = np.dot(X_train.T, hidden_error) / m
            db_encoder = np.sum(hidden_error, axis=0, keepdims=True) / m

            # Mise à jour des poids
            self.W_decoder -= self.learning_rate * dW_decoder
            self.b_decoder -= self.learning_rate * db_decoder
            self.W_encoder -= self.learning_rate * dW_encoder
            self.b_encoder -= self.learning_rate * db_encoder

            # Validation
            if X_val is not None:
                _, val_decoded = self.forward(X_val)
                val_loss = self.compute_loss(X_val, val_decoded)
                self.val_loss_history.append(val_loss)

            # Affichage du progrès
            if verbose and (epoch + 1) % 10 == 0:
                val_msg = f", Val Loss: {val_loss:.6f}" if X_val is not None else ""
                print(f"   Epoch {epoch + 1:3d}/{self.epochs} - Train Loss: {train_loss:.6f}{val_msg}")

        if verbose:
            print("✅ Entraînement terminé!")
        return self

    def compress(self, X):
        """Compression: équivalent à encode"""
        return self.encode(X)

    def decompress(self, encoded):
        """Décompression: équivalent à decode"""
        return self.decode(encoded)

    def reconstruct(self, X):
        """Reconstruction complète"""
        _, decoded = self.forward(X)
        return decoded