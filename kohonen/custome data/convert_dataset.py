# convert_strokes_to_vector_png.py

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==============================================================================
# 1. Fonctions de Chargement et de Tracé des Traits
# ==============================================================================

def load_raw_stroke_data(path, max_samples_per_class=5000, max_classes=10):
    """
    Charge les séquences de traits brutes depuis les fichiers .npz de QuickDraw.
    Retourne une liste de séquences de traits et leurs étiquettes.
    """
    all_strokes = []
    all_labels = []
    class_names = []
    
    try:
        npz_files = [f for f in os.listdir(path) if f.startswith('sketchrnn_') and f.endswith('.full.npz')]
    except FileNotFoundError:
        print(f"ERREUR : Le dossier '{path}' n'a pas été trouvé.")
        return None, None, None
        
    npz_files = sorted(npz_files)[:max_classes]
    
    if not npz_files:
        print(f"ERREUR : Aucun fichier 'sketchrnn_...' trouvé dans '{path}'.")
        return None, None, None

    print(f"Chargement des données de traits brutes pour {len(npz_files)} classes...")

    for label_index, filename in enumerate(npz_files):
        class_name = filename.replace('sketchrnn_', '').replace('.full.npz', '')
        class_names.append(class_name)
        
        data = np.load(os.path.join(path, filename), encoding='latin1', allow_pickle=True)
        strokes = data['train'][:max_samples_per_class]
        
        all_strokes.extend(strokes)
        all_labels.extend([label_index] * len(strokes))

    return all_strokes, np.array(all_labels), class_names

def plot_stroke(ax, stroke_sequence):
    """
    Trace une seule séquence de traits bruts sur un objet Axes de Matplotlib.
    """
    x, y = 0, 0
    current_line_points = []
    
    for dx, dy, pen_state in stroke_sequence:
        current_line_points.append((x, y))
        x += dx
        y += dy
        
        if pen_state == 1: # Le stylo est levé
            points = np.array(current_line_points)
            if len(points) > 1: # S'assurer qu'il y a une ligne à tracer
                ax.plot(points[:, 0], points[:, 1], color='black', linewidth=2)
            current_line_points = []
            
    # Tracer le dernier segment s'il en reste un
    if len(current_line_points) > 1:
        points = np.array(current_line_points)
        ax.plot(points[:, 0], points[:, 1], color='black', linewidth=2)

    # --- Mise en forme ---
    ax.axis('off')
    ax.set_aspect('equal')
    ax.invert_yaxis()

# ==============================================================================
# 2. Script Principal de Conversion en PNG Vectoriels
# ==============================================================================

def convert_strokes_to_vector_png(input_data_folder="quickdraw_data", 
                                  output_parent_folder="quickdraw_vector_pngs",
                                  max_samples_per_class=5000, 
                                  max_classes=10):
    """
    Charge les fichiers .npz de QuickDraw, trace chaque dessin au trait
    et sauvegarde chaque tracé en tant que fichier PNG.
    """
    # --- Configuration des chemins de manière robuste ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, input_data_folder)
    output_path = os.path.join(script_dir, output_parent_folder)
    
    print(f"Dossier d'entrée : {input_path}")
    print(f"Dossier de sortie : {output_path}")
    
    os.makedirs(output_path, exist_ok=True)

    # --- Chargement de toutes les données en une seule fois ---
    all_strokes, all_labels, class_names = load_raw_stroke_data(
        input_path, 
        max_samples_per_class=max_samples_per_class, 
        max_classes=max_classes
    )
    
    if all_strokes is None: return

    print(f"\nDébut de la conversion de {len(all_strokes)} dessins en fichiers PNG...")

    # --- Boucle sur chaque classe pour créer les dossiers ---
    for i, class_name in enumerate(class_names):
        print(f"\nTraitement de la classe : {class_name.capitalize()}")
        class_output_dir = os.path.join(output_path, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Trouver les indices des dessins pour la classe actuelle
        class_indices = np.where(all_labels == i)[0]
        
        # --- Boucle sur chaque dessin de la classe ---
        for drawing_index, global_index in enumerate(tqdm(class_indices, desc=f"  Conversion de {class_name}")):
            stroke_sequence = all_strokes[global_index]
            
            # Créer une nouvelle figure pour chaque dessin
            fig, ax = plt.subplots(figsize=(4, 4)) # Taille de l'image en pouces
            
            # Tracer le dessin
            plot_stroke(ax, stroke_sequence)
            
            # Créer le nom de fichier et le chemin de sauvegarde
            png_filename = f"{class_name}_{drawing_index+1:05d}.png"
            save_path = os.path.join(class_output_dir, png_filename)
            
            # Sauvegarder la figure avec une bonne résolution et en rognant les bords blancs
            fig.savefig(save_path, dpi=96, bbox_inches='tight', pad_inches=0.1)
            
            # TRÈS IMPORTANT : Fermer la figure pour libérer la mémoire
            plt.close(fig)
            
    print("\nConversion terminée avec succès !")

if __name__ == "__main__":
    convert_strokes_to_vector_png(
        input_data_folder="quickdraw_data",
        output_parent_folder="quickdraw_vector_pngs",
        max_samples_per_class=5, # Ajustez pour convertir plus ou moins d'images par classe
        max_classes=15              # Ajustez pour traiter plus ou moins de classes
    )