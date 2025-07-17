# view_raw_strokes.py

import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 1. Raw Stroke Data Loading Function
# ==============================================================================

def load_raw_stroke_data(path, max_samples_per_class=10, max_classes=5):
    """
    Loads raw stroke sequences from QuickDraw sketchrnn .npz files.
    
    Returns a list of stroke arrays, as they have variable lengths.
    """
    all_strokes = []
    all_labels = []
    class_names = []
    
    # --- Correctly set up the absolute path to the data directory ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, path)
    
    try:
        npz_files = [f for f in os.listdir(data_path) if f.startswith('sketchrnn_') and f.endswith('.full.npz')]
    except FileNotFoundError:
        print(f"ERROR: The directory '{data_path}' was not found.")
        print("Please make sure it exists and is in the same directory as this script.")
        return None, None, None
        
    npz_files = sorted(npz_files)[:max_classes]
    
    if not npz_files:
        print(f"ERROR: No 'sketchrnn_...' files found in '{data_path}'.")
        return None, None, None

    print(f"Loading {max_samples_per_class} raw stroke samples from {len(npz_files)} classes...")

    for label_index, filename in enumerate(npz_files):
        class_name = filename.replace('sketchrnn_', '').replace('.full.npz', '')
        class_names.append(class_name.capitalize())
        
        # Load the stroke data
        data = np.load(os.path.join(data_path, filename), encoding='latin1', allow_pickle=True)
        strokes = data['train'][:max_samples_per_class]
        
        all_strokes.extend(strokes)
        all_labels.extend([label_index] * len(strokes))

    print(f"Dataset loaded successfully: {len(all_strokes)} total samples.")
    return all_strokes, np.array(all_labels), class_names

# ==============================================================================
# 2. Plotting and Main Script
# ==============================================================================

def plot_stroke(ax, stroke_sequence):
    """
    Plots a single raw stroke sequence on a matplotlib Axes object.
    
    A stroke sequence is an array of [dx, dy, pen_state].
    pen_state is 0 for "pen down" and 1 for "pen up".
    """
    x, y = 0, 0
    current_line_points = []
    
    for dx, dy, pen_state in stroke_sequence:
        # Add the current point to the line segment
        current_line_points.append((x, y))
        
        # Move the pen
        x += dx
        y += dy
        
        # If the pen is lifted, plot the segment we've collected and reset
        if pen_state == 1:
            points = np.array(current_line_points)
            ax.plot(points[:, 0], points[:, 1], color='black', linewidth=1.5)
            current_line_points = []
            
    # Plot any remaining segment after the loop finishes
    if current_line_points:
        points = np.array(current_line_points)
        ax.plot(points[:, 0], points[:, 1], color='black', linewidth=1.5)

    # --- Formatting ---
    ax.axis('off')
    ax.set_aspect('equal') # Crucial to prevent distortion
    ax.invert_yaxis() # Puts (0,0) at the top-left, like most screen coordinates


def view_raw_quickdraw_strokes():
    """
    Main function to load and display raw QuickDraw stroke data.
    """
    # --- Load Data ---
    # The path is relative to the script's location
    all_strokes, y, class_names = load_raw_stroke_data(
        path='quickdraw_data', 
        max_samples_per_class=10, 
        max_classes=5
    )
    
    if all_strokes is None:
        return

    # --- Create the Plot ---
    num_classes = len(class_names)
    samples_per_class = 10
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(15, 8))
    
    if num_classes == 1: # Handle case of single class to avoid indexing errors
        axes = np.array([axes])
        
    fig.suptitle("Raw QuickDraw Stroke Data (Unprocessed)", fontsize=20)
    
    for i in range(num_classes):
        # Find the indices for the current class
        class_indices = np.where(y == i)[0][:samples_per_class]
        
        # Set the class name as the y-axis label for the first column
        axes[i, 0].set_ylabel(class_names[i], fontsize=14, rotation=0, labelpad=40, ha='right')

        for j, sample_index in enumerate(class_indices):
            stroke_sequence = all_strokes[sample_index]
            plot_stroke(axes[i, j], stroke_sequence)
            
    # Turn off axes for any empty subplots
    for ax in axes.flat:
        if not ax.lines:
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    print("\nDisplaying plot. Close the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    view_raw_quickdraw_strokes()