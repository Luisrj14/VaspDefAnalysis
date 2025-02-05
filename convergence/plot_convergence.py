import sys
import numpy as np
import matplotlib.pyplot as plt
import os

# Add path to my package (ensure you add the directory, not the file itself)
sys.path.append(os.path.abspath('VaspDefAnalysis/convergence'))

from utils_convergence import ConvergenceTools

class ConvergencePlot(ConvergenceTools):
    def __init__(self, 
                 title_name: str,
                 axis_x_name: str,
                 axis_y_name: str = r'$|\Delta E|$ [eV/#atoms]', 
                 difference_energies_unit_meV = False
                 ):
        """Initialize the plot with labels and optional save flag."""
        self.difference_energies_unit_meV = difference_energies_unit_meV
        # Ensure all names are strings
        if isinstance(title_name, str) and isinstance(axis_y_name, str) and isinstance(axis_x_name, str):
            self.title_name = title_name
            self.axis_y_name = axis_y_name
            self.axis_x_name = axis_x_name
        else:
            raise ValueError(f'{title_name}, {axis_y_name}, and {axis_x_name} must all be strings.')

    def plot_convergence(self,
                         difference_energy_values: np.ndarray,
                         cut_off_values: np.ndarray,
                         conv_criterion: float,
                         color_settings: dict = None,
                         axhspan_settings: dict = None,
                         show_fill: bool = True,
                         y_log:bool=True,
                         figure_size: tuple = (6.5, 5)
                         ):
        """
        Plot the convergence data with relative energies vs cut-off values.

        Parameters:
        difference_energy_values (np.ndarray): Relative energies for plotting.
        cut_off_values (np.ndarray): The cut-off values for plotting.
        conv_criterion (float): The convergence threshold.
        color_settings (dict): Optional dictionary to configure plot colors.
        axhspan_settings (dict): Optional settings for the shaded region.
        show_fill (bool): Whether to display the shaded area under the convergence criterion.
        y_log (bool): Whether to use a logarithmic scale for the y-axis.
        figsize (tuple): Optional figure size (width, height) in inches.
        """
        # Default color settings
        default_colors = {
            "line": "black",
            "marker": "o",
            "criterion_line": "red",
            "criterion_style": ":",
            "title_color": "black",
        }
        # Update defaults with user-provided settings
        if color_settings:
            default_colors.update(color_settings)

        # Default axhspan settings
        default_axhspan_settings = {
            "color": "grey",
            "alpha": 0.3
        }
        if axhspan_settings:
            default_axhspan_settings.update(axhspan_settings)

        # Find the convergence values
        cut_off_values_new, convergence_cut_off_values = super().find_convergence_values(difference_energy_values, cut_off_values, conv_criterion)

        if self.difference_energies_unit_meV: 
            # Convert relative energies to [meV]
            difference_energy_values = (10**3) * difference_energy_values
            conv_criterion = (10**3) * conv_criterion
            self.axis_y_name = r'$|\Delta E|$ [meV/#atoms]'
        
        # Set the figure size, use default if None
        plt.figure(figsize=figure_size)

        # Plot data
        if y_log:
            plt.semilogy(
                cut_off_values_new, 
                difference_energy_values,
                color=default_colors["line"], 
                marker=default_colors["marker"], 
                label=r'$|\Delta E|$'
            )
        else:
            plt.plot(
                cut_off_values_new, 
                difference_energy_values,
                color=default_colors["line"], 
                marker=default_colors["marker"], 
                label=r'$|\Delta E|$'
            )

        # Conditionally display the shaded region below the convergence criterion
        if show_fill:
            plt.axhspan(
                ymin=0,
                ymax=conv_criterion,
                **default_axhspan_settings
                )   
        
        # Plot the convergence criterion line
        plt.axhline(
            y=conv_criterion,
            color=default_colors["criterion_line"],
            linestyle=default_colors["criterion_style"],
            label=f'Criterion {conv_criterion} [meV]' if self.difference_energies_unit_meV else f'Criterion {conv_criterion} [eV]'
        )

        # Add titles and labels
        plt.title(
            self.title_name,
            fontdict={
                "family": "serif",
                "color": default_colors["title_color"],
                "weight": "bold",
                "size": 14
            }
        )
        plt.ylabel(self.axis_y_name)
        plt.xlabel(self.axis_x_name)
        plt.legend(loc='upper right')

        # Print the criterion
        print(
            f"Convergence criterion: {conv_criterion} [meV]" 
            if self.difference_energies_unit_meV 
            else f"Convergence criterion: {conv_criterion} [eV]"
        )
        # Print the convergence cut-off values
        print(f"Convergence values: {convergence_cut_off_values}")

        # Display the plot
        # plt.show()
