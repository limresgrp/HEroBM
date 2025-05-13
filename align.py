import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import os

def load_structures(pdb1_path, pdb2_path):
    """Load two PDB structures using MDAnalysis."""
    try:
        u1 = mda.Universe(pdb1_path)
        u2 = mda.Universe(pdb2_path)
        print(f"Successfully loaded:\n- {os.path.basename(pdb1_path)}\n- {os.path.basename(pdb2_path)}")
        return u1, u2
    except Exception as e:
        print(f"Error loading PDB files: {e}")
        return None, None

def get_sequences(u1, u2):
    """Extract amino acid sequences from protein structures."""
    # Extract sequences from protein structures
    seq1 = [r.resname for r in u1.residues if r.resname in mda.lib.util.canonical_inverse_aa_codes.keys()]
    seq2 = [r.resname for r in u2.residues if r.resname in mda.lib.util.canonical_inverse_aa_codes.keys()]
    
    # Convert 3-letter to 1-letter amino acid codes
    seq1_1letter = ''.join([mda.lib.util.convert_aa_code(aa) for aa in seq1])
    seq2_1letter = ''.join([mda.lib.util.convert_aa_code(aa) for aa in seq2])
    
    return seq1_1letter, seq2_1letter

def align_structures(u1, u2):
    """Perform structural alignment of two protein structures."""
    # Select protein backbone atoms for alignment
    mobile = u1.select_atoms("protein and backbone")
    reference = u2.select_atoms("protein and backbone")
    
    # Check if selections have atoms
    if len(mobile) == 0 or len(reference) == 0:
        print("Warning: No backbone atoms found for alignment")
        return None
    
    # Perform alignment
    try:
        alignment = align.alignto(mobile, reference, weights="mass")
        rmsd = alignment[1]  # Get RMSD value
        print(f"Structural alignment RMSD: {rmsd:.2f} Å")
        return rmsd
    except Exception as e:
        print(f"Error during structural alignment: {e}")
        return None

def simple_sequence_alignment(seq1, seq2):
    """Perform a simple global sequence alignment."""
    # Create a simple scoring matrix (1 for match, 0 for mismatch)
    match_score = np.ones((len(seq1)+1, len(seq2)+1))
    
    # Initialize the matrix
    for i in range(len(seq1)+1):
        match_score[i, 0] = 0
    for j in range(len(seq2)+1):
        match_score[0, j] = 0
    
    # Dynamic programming to fill the matrix
    for i in range(1, len(seq1)+1):
        for j in range(1, len(seq2)+1):
            if seq1[i-1] == seq2[j-1]:
                match_score[i, j] = match_score[i-1, j-1] + 1
            else:
                match_score[i, j] = max(match_score[i-1, j], match_score[i, j-1])
    
    # Traceback to get aligned sequences
    i, j = len(seq1), len(seq2)
    aligned_seq1, aligned_seq2 = "", ""
    
    while i > 0 and j > 0:
        if seq1[i-1] == seq2[j-1]:
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            i -= 1
            j -= 1
        elif match_score[i-1, j] > match_score[i, j-1]:
            aligned_seq1 = seq1[i-1] + aligned_seq1
            aligned_seq2 = "-" + aligned_seq2
            i -= 1
        else:
            aligned_seq1 = "-" + aligned_seq1
            aligned_seq2 = seq2[j-1] + aligned_seq2
            j -= 1
    
    # Handle remaining characters
    while i > 0:
        aligned_seq1 = seq1[i-1] + aligned_seq1
        aligned_seq2 = "-" + aligned_seq2
        i -= 1
    while j > 0:
        aligned_seq1 = "-" + aligned_seq1
        aligned_seq2 = seq2[j-1] + aligned_seq2
        j -= 1
    
    return aligned_seq1, aligned_seq2

def plot_alignment(aligned_seq1, aligned_seq2, output_file=None):
    """Visualize the sequence alignment."""
    # Count matches and calculate identity percentage
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b)
    identity = (matches / len(aligned_seq1)) * 100 if len(aligned_seq1) > 0 else 0
    
    # Create a visualization of the alignment
    alignment_viz = []
    for a, b in zip(aligned_seq1, aligned_seq2):
        if a == b:
            alignment_viz.append(1)  # Match
        elif a == '-' or b == '-':
            alignment_viz.append(0)  # Gap
        else:
            alignment_viz.append(0.5)  # Mismatch
    
    # Plot the alignment
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Create a custom colormap: blue for match, white for gap, red for mismatch
    cmap = LinearSegmentedColormap.from_list('alignment_cmap', 
                                            [(0, 'white'), (0.5, 'red'), (1, 'blue')])
    
    # Plot the alignment as a heatmap
    im = ax.imshow([alignment_viz], aspect='auto', cmap=cmap, vmin=0, vmax=1)
    
    # Add a color bar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', ticks=[0, 0.5, 1])
    cbar.set_ticklabels(['Gap', 'Mismatch', 'Match'])
    
    # Set labels and title
    ax.set_yticks([])
    ax.set_xlabel('Alignment Position')
    ax.set_title(f'Sequence Alignment (Identity: {identity:.1f}%)')
    
    # Save the plot if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    
    return identity

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Align two PDB structures and their sequences')
    parser.add_argument('pdb1', help='Path to the first PDB file')
    parser.add_argument('pdb2', help='Path to the second PDB file')
    parser.add_argument('--output', '-o', help='Output file for alignment visualization (optional)')
    args = parser.parse_args()
    
    # Load structures
    u1, u2 = load_structures(args.pdb1, args.pdb2)
    if u1 is None or u2 is None:
        return
    
    # Extract sequences
    seq1, seq2 = get_sequences(u1, u2)
    print(f"\nSequence 1 ({len(seq1)} residues): {seq1[:50]}..." if len(seq1) > 50 else f"\nSequence 1 ({len(seq1)} residues): {seq1}")
    print(f"Sequence 2 ({len(seq2)} residues): {seq2[:50]}..." if len(seq2) > 50 else f"Sequence 2 ({len(seq2)} residues): {seq2}")
    
    # Perform sequence alignment
    print("\nPerforming sequence alignment...")
    aligned_seq1, aligned_seq2 = simple_sequence_alignment(seq1, seq2)
    
    # Display alignment
    print("\nAlignment:")
    for i in range(0, len(aligned_seq1), 60):
        segment1 = aligned_seq1[i:i+60]
        segment2 = aligned_seq2[i:i+60]
        match_line = ''.join('|' if a == b else ' ' for a, b in zip(segment1, segment2))
        print(f"{segment1}")
        print(f"{match_line}")
        print(f"{segment2}")
        print("")
    
    # Plot alignment
    identity = plot_alignment(aligned_seq1, aligned_seq2, args.output)
    print(f"Sequence identity: {identity:.1f}%")
    
    # Structural alignment
    print("\nPerforming structural alignment...")
    align_structures(u1, u2)

if __name__ == "__main__":
    main()