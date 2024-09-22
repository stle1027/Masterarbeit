import torch
import torch.nn.functional as F
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

class PrototypeSimilarity:
    def __init__(self, mode):
        self.mode = mode

    def load_prototypes(self, file_path):
        """Load prototypes from a file."""
        return torch.load(file_path)

    def calculate_cosine_similarities(self, prototypes_a, prototypes_b=None):
        """Calculate cosine similarities between two sets of prototypes."""
        if prototypes_b is None:
            prototypes_b = prototypes_a
        return F.cosine_similarity(prototypes_a.unsqueeze(1), prototypes_b.unsqueeze(0), dim=2)

    def check_similarity(self, similarities, labels, threshold=0.7):
        """Check for pairs with similarity above the threshold."""
        warnings = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if similarities[i, j] > threshold:
                    warnings.append((labels[i], labels[j], similarities[i, j]))
        return warnings

    def plot_global_similarity(self, prototypes, labels):
        """Plot global similarity heatmap and display warnings."""
        similarities = self.calculate_cosine_similarities(prototypes)
        similarity_df = pd.DataFrame(similarities.numpy(), index=labels, columns=labels)

        # Plot setup
        figsize = (15, 8)
        fig, ax = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 1]})

        # Heatmap
        im = ax[0].imshow(similarity_df, cmap="viridis", vmin=0, vmax=1)

        # Colorbar
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)

        # Axis labels
        ax[0].set_xticks(np.arange(similarity_df.shape[1]))
        ax[0].set_yticks(np.arange(similarity_df.shape[0]))
        ax[0].set_xticklabels([label.replace('_', '\n') for label in similarity_df.columns], ha="center", fontsize=12)
        ax[0].set_yticklabels([label.replace('_', '\n') for label in similarity_df.index], va="center", fontsize=12)

        # Annotations
        for i in range(similarity_df.shape[0]):
            for j in range(similarity_df.shape[1]):
                if i < j:  # Right of the diagonal
                    ax[0].add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color='black', lw=0))
                else:  # Diagonal and left of it
                    ax[0].text(j, i, f'{similarity_df.iat[i, j]:.3f}', ha='center', va='center', color='black', fontsize=12)

        ax[0].grid(False)
        ax[0].tick_params(top=False, bottom=False, left=False, right=False)
        ax[0].set_title("Part Similarity Heatmap", fontsize=12)

        # Check and display warnings
        warnings = self.check_similarity(similarities, labels)
        if warnings:
            text_output = []
            base_path = "prototypes/parts/all/"
            for w in warnings:
                warning_text = f"Similarity between '{w[0]}' and '{w[1]}' is {w[2]:.2f}. This is above the threshold of 0.8.\n"
                text_output.append(warning_text)
                for class_name in [w[0], w[1]]:
                    try:
                        class_path = os.path.join(base_path, class_name)
                        files = os.listdir(class_path)
                        seiten = [file.replace(f"{class_name}_", "").replace(".pth", "") for file in files if file.startswith(class_name)]
                        seiten.sort()
                    except:
                        None
                    try:
                        text_output.append(f"Sides for class '{class_name}': {', '.join(seiten)}\n")
                    except:
                        None

            text_output_str = ''.join(text_output)  # Combine all warning texts into one string
        else:
            text_output_str = "Keine Paare von Hauptprototypen haben eine Ähnlichkeit über 80%.\n"

        # Wrapping the text for better display in the plot
        wrapped_text = '\n'.join([text_output_str[i:i+200] for i in range(0, len(text_output_str), 200)])
        ax[1].text(0, 0.85, wrapped_text, ha="left", va="top", fontsize=10, wrap=True, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_class_similarity(self, class1, side1, class2, side2):
        """Plot similarity between two classes."""
        parts_files = {
            "class1": f"prototypes/parts/all/{class1}/{class1}_{side1}.pth",
            "class2": f"prototypes/parts/all/{class2}/{class2}_{side2}.pth"
        }

        class1_data = self.load_prototypes(parts_files["class1"])
        class2_data = self.load_prototypes(parts_files["class2"])

        class1_filtered_indices = [i for i, label in enumerate(class1_data['label_names']) if not label.startswith("COLOR")]
        class2_filtered_indices = [i for i, label in enumerate(class2_data['label_names']) if not label.startswith("COLOR")]

        class1_prototypes = class1_data['prototypes'][class1_filtered_indices]
        class2_prototypes = class2_data['prototypes'][class2_filtered_indices]

        class1_labels = [f"{label} ({class1})" for label in class1_data['label_names'] if not label.startswith("COLOR")]
        class2_labels = [f"{label} ({class2})" for label in class2_data['label_names'] if not label.startswith("COLOR")]

        cross_similarities = self.calculate_cosine_similarities(class1_prototypes, class2_prototypes)
        cross_similarity_df = pd.DataFrame(cross_similarities.numpy(), index=class1_labels, columns=class2_labels)

        # Plotting the Heatmap
        fig, ax = plt.subplots(figsize=(15, 8))
        cax = ax.imshow(cross_similarity_df, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(cax, ax=ax)

        ax.set_xticks(np.arange(cross_similarity_df.shape[1]))
        ax.set_yticks(np.arange(cross_similarity_df.shape[0]))
        ax.set_yticklabels([self.format_label(label) for label in cross_similarity_df.index], va="center", ha="right", fontsize=12)
        ax.set_xticklabels([self.format_label(label) for label in cross_similarity_df.columns], ha="center", va="top", fontsize=12)

        for i in range(cross_similarity_df.shape[0]):
            for j in range(cross_similarity_df.shape[1]):
                ax.text(j, i, f'{cross_similarity_df.iat[i, j]:.4f}', ha='center', va='center', color='black', fontsize=12)

        ax.grid(False)
        ax.tick_params(top=False, bottom=False, left=False, right=False)

        plt.title(f"Part Similarity Heatmap - {class1} vs {class2}")
        plt.show()

    def format_label(self, label):
        """Format label for better visualization."""
        if ' (' in label:
            return label.replace(' (', '\n(')
        return label

    def run(self):
        """Run the appropriate mode based on user input."""
        if self.mode == "global":
            global_prototypes_data = self.load_prototypes("ycb_prototypes.pth")
            self.plot_global_similarity(global_prototypes_data['prototypes'], global_prototypes_data['label_names'])
        else:
            class1 = str(input("Class name 1: "))
            side1 = str(input("Side name for class 1: "))
            class2 = str(input("Class name 2: "))
            side2 = str(input("Side name for class 2: "))
            self.plot_class_similarity(class1, side1, class2, side2)

if __name__ == "__main__":
    x = str(input("Check for local or global similarities? (enter 'global' or 'local'): "))
    prototype_similarity = PrototypeSimilarity(x)
    prototype_similarity.run()
