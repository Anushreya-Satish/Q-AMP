import matplotlib.pyplot as plt
import numpy as np


def parse_sequence(sequence_str: str):
    if isinstance(sequence_str, (list, tuple)):
        return list(sequence_str)
    return sequence_str.split("-")


def plot_results(sequence: str, filename: str = "qaoa_amp_results.png"):
    residues = parse_sequence(sequence)
    n_res = len(residues)

    hydropathy_map = {"Leu": 3.8, "Lys": -3.9}
    charge_map = {"Leu": 0, "Lys": 1}

    hydropathy = [hydropathy_map.get(res, 0.0) for res in residues]
    charges = [charge_map.get(res, 0) for res in residues]
    positions = np.arange(1, n_res + 1)

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"Q-AMP Structural & Biophysical Profile (N = {n_res})", fontsize=16, fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(1, 2, 1)
    colors = ['#2ca02c' if h > 0 else '#1f77b4' for h in hydropathy]
    
    bars = ax1.bar(positions, hydropathy, color=colors, alpha=0.85, edgecolor='black', width=0.6)
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    ax1.set_title("Residue Hydropathy Profile (Kyte-Doolittle)", fontsize=12, pad=10)
    ax1.set_xlabel("Residue Position", fontsize=11)
    ax1.set_ylabel("Hydropathy Score (+ Hydrophobic / - Hydrophilic)", fontsize=10)
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f"{r}\n({i})" for i, r in enumerate(residues, 1)], fontsize=9)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        va = 'bottom' if yval >= 0 else 'top'
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:+.1f}", ha='center', va=va, fontsize=8, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', edgecolor='black', label='Leucine (Hydrophobic / Membrane Anchor)'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='Lysine (Cationic / Cell Target)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2, projection='polar')
    
    angles = np.radians(np.array([(i * 100) % 360 for i in range(n_res)]))
    radii = np.linspace(0.4, 1.0, n_res)

    for i, (angle, r, res) in enumerate(zip(angles, radii, residues)):
        color = '#2ca02c' if res == 'Leu' else '#1f77b4'
        ax2.scatter(angle, r, color=color, s=400, edgecolors='black', zorder=3)
        ax2.text(angle, r, f"{res}\n{i+1}", ha='center', va='center', color='white', fontweight='bold', fontsize=8, zorder=4)

    for i in range(n_res - 1):
        ax2.plot([angles[i], angles[i+1]], [radii[i], radii[i+1]], color='gray', linestyle='-', linewidth=1.5, alpha=0.7, zorder=2)

    ax2.set_title("2D Alpha-Helical Wheel Projection", fontsize=12, pad=15)
    ax2.set_yticklabels([])
    ax2.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("--- Running standalone verification for visualization.py ---")
    test_seq = "Leu-Lys-Leu-Lys-Leu-Lys"
    plot_results(test_seq, "test_qaoa_amp_results.png")
    print("Successfully generated 'test_qaoa_amp_results.png'.")