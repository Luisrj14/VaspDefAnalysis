#!/usr/bin/env python3

"""
Script to plot localized states using VaspDefAnalysis.
"""

import matplotlib.pyplot as plt
from VaspDefAnalysis.plotter.plot_localization_factor import PlotLocalizedStates
import argparse


def main(
    perfect_structure_path: str,
    defect_structure_path: str,
    defect_vasprun_path: str,
    VBM: float,
    CBM: float,
    y_limit: tuple | None = "(VBM-1.5,CBM+1.5)",  # Default is None
    fermi_energy_reference: bool = True,
    show_fill_up: bool = True,
    tolerance: float = 0.1,
    add_neighbors_up: int = 1,
    norm: bool = True,
    **plot_settings,
):
    """
    Main function to generate plots for localized states.

    Args:
        perfect_structure_path (str): Path to the perfect structure POSCAR.
        defect_structure_path (str): Path to the defect structure POSCAR.
        defect_vasprun_path (str): Path to the defect vasprun.xml.
        VBM (float): Valence band maximum energy.
        CBM (float): Conduction band minimum energy.
        y_limit (tuple | None): y-axis limits for the plot, by default is None.
        fermi_energy_reference (bool): Whether to reference the Fermi energy.
        show_fill_up (bool): Whether to show filled states.
        tolerance (float): Tolerance level for plotting.
        add_neighbors_up (int): Number of neighbors to add above the band.
        norm (bool): Normalize the data.
        **plot_settings: Additional plotting settings.
    """

    # Plot localized states
    fig_1 = PlotLocalizedStates.get_plot_LS_using_IOWs(
        perfect_structure_path=perfect_structure_path,
        defect_structure_path=defect_structure_path,
        defect_vasprun_path=defect_vasprun_path,
        VBM=VBM,
        CBM=CBM,
        y_limit=y_limit,
        fermi_energy_reference=fermi_energy_reference,
        show_fill_up=show_fill_up,
        tolerance=tolerance,
        add_neighbors_up=add_neighbors_up,
        norm=norm,
        **plot_settings,
    )
    # Activate and display the figure using plt.figure()
    fig_1.savefig(fname="kS_LS_values")

    # Example for Kohn-Sham eigenvalues plot (uncomment if needed)
    # fig_2 = PlotKohnShamEigenvalue.get_plot_KS(
    #     vasprun_path=defect_vasprun_path,
    #     VBM=VBM,
    #     CBM=CBM,
    #     **plot_settings
    # )
    # plt.show(fig_2)


if __name__ == "__main__":
    # Set up argument parsing for flexibility
    parser = argparse.ArgumentParser(
        description="Generate plots for localized states and Kohn-Sham eigenvalues."
    )
    parser.add_argument(
        "--perfect_path",
        type=str,
        required=True,
        help="Path to the perfect structure POSCAR.",
    )
    parser.add_argument(
        "--defect_path",
        type=str,
        required=True,
        help="Path to the defect structure POSCAR.",
    )
    parser.add_argument(
        "--vasprun_path",
        type=str,
        required=True,
        help="Path to the defect vasprun.xml.",
    )
    parser.add_argument("--VBM", type=float, required=True, help="Valence Band Maximum energy.")
    parser.add_argument("--CBM", type=float, required=True, help="Conduction Band Minimum energy.")
    parser.add_argument("--y_limit", type=tuple, default="(VBM-1.5,CBM+1.5)", help="y-axis limits as a tuple (-1,9)")

    parser.add_argument(
        "--fermi_energy_reference",
        action="store_true",
        help="Reference the Fermi energy.",
        default=True,
    )
    parser.add_argument(
        "--show_fill_up",
        action="store_true",
        help="Show filled states in the plot.",
        default=True,
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.1, help="Tolerance level for plotting."
    )
    parser.add_argument(
        "--add_neighbors_up", type=int, default=1, help="Number of neighbors to add above the band."
    )
    parser.add_argument("--norm", action="store_true", help="Normalize the data.", default=True)

    args = parser.parse_args()

    main(
        perfect_structure_path=args.perfect_path,
        defect_structure_path=args.defect_path,
        defect_vasprun_path=args.vasprun_path,
        VBM=args.VBM,
        CBM=args.CBM,
        y_limit=args.y_limit,
        fermi_energy_reference=args.fermi_energy_reference,
        show_fill_up=args.show_fill_up,
        tolerance=args.tolerance,
        add_neighbors_up=args.add_neighbors_up,
        norm=args.norm,
    )


