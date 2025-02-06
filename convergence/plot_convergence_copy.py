import sys
import numpy as np
import matplotlib.pyplot as plt
import os

# Add path to my package (ensure you add the directory, not the file itself)
sys.path.append(os.path.abspath('VaspDefAnalysis/convergence'))

from VaspDefAnalysis.convergence.utils_convergence import ConvergenceTools

class ConvergencePlot(ConvergenceTools):
    def __init__(self, 
                 title_name: str,
                 axis_x_name: str,
                 axis_y_name: str, 
                 label_relative_values:str,
                 label_conv_criterion:str
                 ):
        """Initialize the plot with labels and axis names."""
        self.title_name = title_name
        self.axis_y_name = axis_y_name
        self.axis_x_name = axis_x_name
        self.label_relative_values = label_relative_values
        self.label_conv_criterion = label_conv_criterion

    def plot_setting_kwargs(self, **plot_setting):
        """
        Handle and validate keyword arguments for plot customization.

        Parameters:
        ----------
        plot_setting : dict
            Arbitrary keyword arguments for plot customization.

        Returns:
        -------
        dict
            A dictionary of standardized plot settings.
        """
        # Define default settings
        default_settings = {
        "fontdict_title": {"family": "serif","color": "black","weight": "bold","size": 14},
        "label_size": 12,
        "figsize": (6,6), 
        "criterion_settings": {"linestyle":":","color":"red"},
        "legend_loc": "upper right",
        "fill_settings": {"color": "grey","alpha": 0.3}
        }
        
        # Validate keys
        valid_keys = default_settings.keys()
        invalid_keys = [key for key in plot_setting if key not in valid_keys]
        if invalid_keys:
            raise ValueError(f"Invalid keys in plot_setting: {invalid_keys}")
    
        # Separate out dictionary-based settings to be updated
        dict_keys = ['fontdict_title']
        for dict_key in dict_keys:
            if dict_key in plot_setting:
                # Update the existing dictionary with new settings from plot_setting
                plot_setting[dict_key] = {**default_settings[dict_key], **plot_setting[dict_key]}
    
        # Update default settings with user-provided settings
        validated_settings = {**default_settings, **plot_setting}
        return validated_settings
    
    def plot_convergence(self,
                         relative_values: np.ndarray,
                         cutoff_values: np.ndarray,
                         conv_criterion: float,
                         show_fill: bool = True,
                         y_log:bool=True,
                         **settings
                         ):
        """
        Plot the convergence data with relative energies vs cut-off values.

        Parameters:
        relative_values (np.ndarray): Relative energies for plotting.
        cutoff_values (np.ndarray): The cut-off values for plotting.
        conv_criterion (float): The convergence threshold.
        color_settings (dict): Optional dictionary to configure plot colors.
        axhspan_settings (dict): Optional settings for the shaded region.
        show_fill (bool): Whether to display the shaded area under the convergence criterion.
        y_log (bool): Whether to use a logarithmic scale for the y-axis.
        figsize (tuple): Optional figure size (width, height) in inches.
        """

        # Handle plot settings
        plot_settings = self.plot_setting_kwargs(**settings)

        # Instance the subplot class's matplotlib
        fig = plt.subplot(1,1,figsize=(plot_settings["figsize"]))

        # Find the convergence values
        cutoff_values_new, convergence_cutoff_values = super().find_convergence_values(relative_values, cutoff_values, conv_criterion)

        # Plot x,y velues
        if y_log: 
            fig.semilogy(cutoff_values_new,relative_values,label=self.label_relative_values)
        else:
            fig.plot(cutoff_values_new,relative_values,label=self.label_relative_values)
         
        # Plot the convergence criterion line
        fig.axhline(
                y=conv_criterion,
                label=self.label_conv_criterion,
                **plot_settings['criterion_settings']
            )
        
        # Conditionally display the shaded region below the convergence criterion
        if show_fill:
            fig.axhspan(
                ymin=0,
                ymax=conv_criterion,
                **plot_settings["fill_settings"]
                )
        
        # Add titles and labels
        fig.title(
            self.title_name,
            fontdict=plot_settings['fontdict_title']
        )
        fig.ylabel(self.axis_y_name)
        fig.xlabel(self.axis_x_name)
        fig.legend(loc=plot_settings["legend_loc"])  

        return fig 
    
    @staticmethod
    def get_energy_convergence_plot(energies:np.ndarray,
                                    cut_off_values:np.ndarray,
                                    conv_criterion:float,
                                    title_name: str= "Total energy vs Cutoff energy",
                                    axis_x_name: str="Cutoff [eV]",
                                    axis_y_name: str= r'$|\Delta E|$ [meV/#atoms]', 
                                    label_relative_values:str = r'$|\Delta E|$',
                                    label_conv_criterion_unit:str="meV",
                                    show_fill:bool=True,
                                    y_log:bool=True,
                                    **settings
                                    ):
        Fig_energy = ConvergencePlot(title_name=title_name,axis_x_name=axis_x_name,axis_y_name=axis_y_name,label_conv_criterion=label_relative_values,label_relative_values=f'Criterion {conv_criterion} {label_conv_criterion_unit}')
        
        relative_energies = Fig_energy.create_abs_diff_values(energies)

        conv_energy_plot = Fig_energy.plot_convergence(relative_values=relative_energies,
                                                       cutoff_values=cut_off_values,
                                                       conv_criterion=conv_criterion,
                                                       show_fill=show_fill,
                                                       y_log=y_log,
                                                       **settings)
        return conv_energy_plot