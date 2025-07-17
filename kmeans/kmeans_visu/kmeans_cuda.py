# kmeans_cuda_fr.py
import numpy as np
import torch
from tqdm import tqdm

class KMeansCUDA:
    def __init__(self, n_clusters: int, max_iter: int = 300,
                 tol: float = 1e-4, device: str | torch.device = 'cuda'):
        """
        Args:
            n_clusters (int): Le nombre de clusters (groupes) à créer (k).
            max_iter (int): Le nombre maximum d'itérations pour l'algorithme.
            tol (float): La tolérance pour la convergence. Si les centroïdes bougent
                         moins que cette valeur, l'algorithme s'arrête.
            device (str | torch.device): Le périphérique à utiliser ('cuda' ou 'cpu').
        """
        self.n_clusters = n_clusters
        self.max_iter   = max_iter
        self.tol        = tol
        
        # Détection automatique du périphérique avec basculement sur le CPU si CUDA n'est pas disponible
        if isinstance(device, str):
            if device == 'cuda' and not torch.cuda.is_available():
                print("CUDA non disponible, basculement sur le CPU")
                device = 'cpu'
            self.device = torch.device(device)
        else:
            self.device = device
            
        print(f"Utilisation de : {self.device}")
        
        # Les centroïdes finaux (les centres des clusters)
        self.centroides: torch.Tensor | None = None
        # Les étiquettes (l'ID du cluster pour chaque point de donnée)
        self.etiquettes: torch.Tensor | None = None
    
    def fit(self, X: np.ndarray):
        """
        Entraîne le modèle K-Moyennes sur les données fournies.

        Args:
            X (np.ndarray): Le tableau NumPy des données d'entraînement.
        
        Returns:
            KMeansCUDA: L'instance de l'objet lui-même, après entraînement.
        """
        # 1. Conversion du tableau NumPy en tenseur PyTorch et envoi sur le bon périphérique (GPU/CPU)
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        
        # 2. Initialisation des centroïdes (méthode "Forgy")
        #    - On mélange aléatoirement les indices des points de données.
        #    - On sélectionne les `n_clusters` premiers points comme centroïdes initiaux.
        #    - .clone() crée une copie pour ne pas modifier les données originales.
        perm = torch.randperm(X.shape[0], device=self.device)
        self.centroides = X[perm[:self.n_clusters]].clone()
        
        # 3. Boucle principale de l'algorithme K-Moyennes
        for it in tqdm(range(self.max_iter), desc="Entraînement K-Moyennes"):
            # --- Étape d'Assignation (E-Step) ---
            # Assigne chaque point de donnée au centroïde le plus proche.
            self.etiquettes = self._assigner_clusters(X)
            
            # --- Étape de Mise à Jour (M-Step) ---
            # Recalcule les centroïdes en fonction des nouvelles assignations.
            nouveaux_centroides = self._mettre_a_jour_centroides(X)
            
            # --- Vérification de la convergence ---
            # On vérifie si les nouveaux centroïdes sont très proches des anciens.
            if torch.allclose(nouveaux_centroides, self.centroides, atol=self.tol):
                print(f"Convergence atteinte à l'itération {it}")
                break # On sort de la boucle
                
            # Si non convergé, on met à jour les centroïdes pour la prochaine itération.
            self.centroides = nouveaux_centroides
        
        return self
    
    def _assigner_clusters(self, X: torch.Tensor) -> torch.Tensor:
        """Étape d'assignation : trouve le centroïde le plus proche pour chaque point."""
        # torch.cdist calcule efficacement la distance entre chaque point de X et chaque centroïde.
        distances = torch.cdist(X, self.centroides)
        
        # Pour chaque point (chaque ligne), on trouve l'indice de la colonne avec la distance minimale.
        # Cet indice est l'ID du cluster le plus proche.
        return torch.argmin(distances, dim=1)
    
    def _mettre_a_jour_centroides(self, X: torch.Tensor) -> torch.Tensor:
        """Étape de mise à jour : recalcule la position de chaque centroïde."""
        liste_centroides = []
        # On parcourt chaque ID de cluster
        for k in range(self.n_clusters):
            # On trouve tous les points qui ont été assignés au cluster 'k'
            points_du_cluster = X[self.etiquettes == k]
            
            # On gère le cas où un cluster pourrait devenir vide
            if len(points_du_cluster) > 0:
                # Le nouveau centroïde est la moyenne de tous les points du cluster.
                liste_centroides.append(points_du_cluster.mean(0))
            else:
                # Si le cluster est vide, on conserve sa position précédente.
                liste_centroides.append(self.centroides[k])
                
        # On empile la liste des tenseurs de centroïdes en un seul tenseur.
        return torch.stack(liste_centroides)
    
    @torch.inference_mode()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit l'ID du cluster pour de nouvelles données, après l'entraînement.
        
        Args:
            X (np.ndarray): Le tableau NumPy des nouvelles données.
        
        Returns:
            np.ndarray: Un tableau d'IDs de cluster pour chaque point de donnée.
        """
        # Conversion des nouvelles données en tenseur sur cpu/gpu
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        
        # On assigne les nouveaux points aux centroïdes finaux (appris)
        # et on convertit le résultat en tableau NumPy pour l'utilisateur.
        return self._assigner_clusters(X).cpu().numpy()