import glob
import os
from typing import Optional
import numpy as np
import MDAnalysis as mda
import seaborn as sns
import pandas as pd

from MDAnalysis.analysis.dihedrals import Ramachandran, Janin
from matplotlib import pyplot as plt


def plot_rama_janin(dih_list, dih_list_true, ref_back: bool = True):
    fig_rama, ax_rama = plt.subplots(figsize=(8, 8), facecolor='white')
    fig_janin, ax_janin = plt.subplots(figsize=(8, 8), facecolor='white')
    
    for (rama, janin) in dih_list:
        rama.plot(ax=ax_rama, ref=ref_back, marker='d', color='red', edgecolors='black', s=25, linewidth=.5)
        janin.plot(ax=ax_janin, ref=ref_back, marker='d', color='red', edgecolors='black', s=25, linewidth=.5)
        
    for (rama_true, janin_true) in dih_list_true:
        rama_true_scatter = rama_true.results.angles.reshape(np.prod(rama_true.results.angles.shape[:2]), 2)
        ax_rama.scatter(rama_true_scatter[:, 0], rama_true_scatter[:, 1], marker='x', s=20, alpha=0.2, color='black', edgecolors='lime', linewidth=2)
        janin_true_scatter = janin_true.results.angles.reshape(np.prod(janin_true.results.angles.shape[:2]), 2)
        ax_janin.scatter(janin_true_scatter[:, 0], janin_true_scatter[:, 1], marker='x', s=20, alpha=0.2, color='black', edgecolors='lime', linewidth=2)
    
    ax_rama.set_xlabel(r'')
    ax_rama.set_ylabel(r'')
    ax_janin.set_xlabel(r'')
    ax_janin.set_ylabel(r'')
    
    ax_rama.set_xticks([-180, -90, 0, 90, 180])
    ax_rama.set_xticklabels([r'-$\pi$', r'-$\pi/2$','0', r'$\pi$/2', r'$\pi$'])
    ax_rama.set_yticks([-180, -90, 0, 90, 180])
    ax_rama.set_yticklabels([r'-$\pi$', r'-$\pi/2$','0', r'$\pi/2$', r'$\pi$'])
    
    ax_janin.set_xticks([0, 90, 180, 270, 360])
    ax_janin.set_xticklabels([r'-$\pi$', r'-$\pi/2$','0', r'$\pi$/2', r'$\pi$'])
    ax_janin.set_yticks([0, 90, 180, 270, 360])
    ax_janin.set_yticklabels([r'-$\pi$', r'-$\pi/2$','0', r'$\pi/2$', r'$\pi$'])
    
    plt.show()
    return fig_rama, fig_janin

def compute_dihedrals(filename: str):
    u = mda.Universe(filename)
    protein = u.select_atoms(f'protein')
    rama = Ramachandran(protein).run()
    protein_chi = u.select_atoms(f'protein and resname ARG ASN ASP GLN GLU HIE HID HIS ILE LEU LYS MET TRP TYR')
    janin = Janin(protein_chi).run()

    return rama, janin

def normalise(x):
    x = x / 180 * np.pi
    if np.any(x > np.pi):
        x -= np.pi
    return x

def plot_distribution(dih_list, dih_list_true, title:str=None, thresh=0.05, ref_thresh=0.05, bins=60, show_chi=True):
        csfont = {'fontname':'Comic Sans MS'}

        fig = plt.figure(figsize=(12, 6), facecolor='white')
        ax1 = plt.subplot(1,2,1)
        if show_chi:
            ax2 = plt.subplot(2,2,2)
            ax3 = plt.subplot(2,2,4)

        plt.subplots_adjust(
            left=0.1,
            bottom=0.1,
            right=0.9,
            top=0.9,
            wspace=0.02,
            hspace=0.02,
        )

        phi_psi = normalise(np.concatenate([dih[0].results.angles.reshape(np.prod(dih[0].results.angles.shape[:2]), 2) for dih in dih_list], axis=0))
        
        if title is not None:
            ax1.set_title(title, **csfont)
        sns.kdeplot(
            x=phi_psi[:, 0],
            y=phi_psi[:, 1],
            cmap=sns.color_palette(f"blend:#EEE,{sns.color_palette().as_hex()[0]}", as_cmap=True),
            fill=True, thresh=thresh, ax=ax1, levels=10, bw=0.18
        )
        sns.kdeplot(
            x=phi_psi[:, 0],
            y=phi_psi[:, 1],
            color=sns.color_palette()[0],
            fill=False, thresh=thresh, ax=ax1, levels=10, linewidths=0.1, bw=0.18
        )

        if dih_list_true is not None and len(dih_list_true) > 0:
            phi_psi_true = normalise(np.concatenate([dih[0].results.angles.reshape(np.prod(dih[0].results.angles.shape[:2]), 2) for dih in dih_list_true], axis=0))
            sns.kdeplot(
                x=phi_psi_true[:, 0],
                y=phi_psi_true[:, 1],
                color=sns.color_palette()[1],
                fill=False, thresh=ref_thresh, ax=ax1, levels=10, linewidths=0.5, bw=0.18
            )
        ax1.set_xlim(xmin=-np.pi, xmax=np.pi)
        ax1.set_ylim(ymin=-np.pi, ymax=np.pi)
        ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax1.set_xticklabels([r'-$\pi$', r'-$\pi/2$','0', r'$\pi$/2', r'$\pi$'])
        ax1.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax1.set_yticklabels([r'-$\pi$', r'-$\pi/2$','0', r'$\pi/2$', r'$\pi$'])

        if show_chi:
            chi1_chi2 = normalise(np.concatenate([dih[1].results.angles.reshape(np.prod(dih[1].results.angles.shape[:2]), 2) for dih in dih_list], axis=0))
            if dih_list_true is not None and len(dih_list_true) > 0:
                chi1_chi2_true = normalise(np.concatenate([dih[1].results.angles.reshape(np.prod(dih[1].results.angles.shape[:2]), 2) for dih in dih_list_true], axis=0))
            else:
                chi1_chi2_true=np.array([[],[]])
        
            chi1_data = np.concatenate([chi1_chi2[:, 0], chi1_chi2_true[:, 0]])
            df = pd.DataFrame(
                {
                    "Angle [rad]": chi1_data,
                    "": np.array(
                        ["Backmapped"]*len(chi1_chi2) + ["Original"]*len(chi1_chi2_true)
                        ),
                }
            )
            sns.histplot(data=df,
                        x="Angle [rad]",
                        bins=bins,
                        hue="",
                        stat = "probability",
                        element="poly",
                        ax=ax2,
            ).set(xlabel=None) # .set_title(r"$\chi1$ Angle Distribution | $\chi2$ Angle Distribution", **csfont)
            ax2.set_xlim(xmin=-np.pi, xmax=np.pi)
            ax2.set_yticks([])
            ax2.set_ylabel('')
            ax2.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax2.set_xticklabels(['', '','', '', ''])

            chi2_data = np.concatenate([chi1_chi2[:, 1], chi1_chi2_true[:, 1]])
            df = pd.DataFrame(
                {
                    "Torsion [rad]": chi2_data,
                    "": np.array(
                        ["Backmapped"]*len(chi1_chi2) + ["Original"]*len(chi1_chi2_true)
                        ),
                }
            )
            sns.histplot(data=df,
                        x="Torsion [rad]",
                        bins=bins,
                        hue="",
                        stat = "probability",
                        element="poly",
                        ax=ax3,
            ).set(xlabel=None)
            ax3.set_xlim(xmin=-np.pi, xmax=np.pi)
            ax3.set_yticks([])
            ax3.set_ylabel('')
            ax3.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
            ax3.set_xticklabels([r'-$\pi$', r'-$\pi/2$','0', r'$\pi/2$', r'$\pi$'])
        plt.show()
        return fig

def analyse_torsions(target: str, ref: Optional[str] = None):

    dih_list = []
    dih_list_true = []

    if os.path.isdir(target):
        target = os.path.join(target, f"*.pdb")

    print("Analysing target structures...")
    for i, trg_filename in enumerate(glob.glob(target)):
        print(f"{i+1} - {trg_filename}")
        rama, janin = compute_dihedrals(trg_filename)
        dih_list.append((rama, janin))

    if ref is not None:
        if os.path.isdir(ref):
            ref = os.path.join(ref, f"*.pdb")

        print("Analysing reference structures...")
        for i, src_filename in enumerate(glob.glob(ref)):
            print(f"{i+1} - {src_filename}")
            rama_true, janin_true = compute_dihedrals(src_filename)
            dih_list_true.append((rama_true, janin_true))
    
    fig_rama, fig_janin = plot_rama_janin(dih_list, dih_list_true, ref_back=True)
    return fig_rama, fig_janin