
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

# Add path to my package (ensure you add the directory, not the file itself)
#sys.path.append(os.path.abspath('VaspDefAnalysis/convergence'))

from VaspDefAnalysis.utils.utils_convergence import ConvergenceTools
from VaspDefAnalysis.utils.tool_pool import SIUnitConverter

class ConvergencePlot(ConvergenceTools):
    def __init__(self, 
                 title_name: str,
                 axis_x_name: str,
                 axis_y_name: str, 
                 label_relative_values: str,
                 label_conv_criterion: str):
        """Initialize the plot with labels and axis names."""
        self.title_name = title_name
        self.axis_x_name = axis_x_name
        self.axis_y_name = axis_y_name
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
        default_settings = {
            "fontdict_title": {"family": "serif", "color": "black", "weight": "bold", "size": 14},
            "label_size": 12,
            "figsize": (6, 5),
            "curve_settings":{"color":"black","marker": ">"},
            "criterion_settings": {"linestyle": ":", "color": "red"},
            "legend_loc": "upper right",
            "fill_settings": {"color": "grey", "alpha": 0.3}
        }
        
        valid_keys = default_settings.keys()
        invalid_keys = [key for key in plot_setting if key not in valid_keys]
        if invalid_keys:
            raise ValueError(f"Invalid keys in plot_setting: {invalid_keys}")
    
        # Update settings with user-provided values
        dict_keys = ['fontdict_title']
        for dict_key in dict_keys:
            if dict_key in plot_setting:
                plot_setting[dict_key] = {**default_settings[dict_key], **plot_setting[dict_key]}
    
        validated_settings = {**default_settings, **plot_setting}
        return validated_settings
    
    def plot_convergence(self,
                         relative_values: np.ndarray,
                         cutoff_values: np.ndarray,
                         conv_criterion: float,
                         show_fill: bool = True,
                         y_log: bool = True,
                         **settings):
        """
        Plot the convergence data with relative energies vs cut-off values.

        Parameters:
        ----------
        relative_values : np.ndarray
            Relative energies for plotting.
        cutoff_values : np.ndarray
            The cut-off values for plotting.
        conv_criterion : float
            The convergence threshold.
        show_fill : bool
            Whether to display the shaded area under the convergence criterion.
        y_log : bool
            Whether to use a logarithmic scale for the y-axis.
        settings : dict
            Additional plot settings.
        
        Returns:
        -------
        fig : matplotlib.figure.Figure
            The figure object created.
        """
        
        #if len(relative_values) != len(cutoff_values):
        #    raise ValueError("Lengths of relative_values and cutoff_values must match.")

        plot_settings = self.plot_setting_kwargs(**settings)
        
        fig, ax = plt.subplots(figsize=plot_settings["figsize"])
        cutoff_values_new, convergence_cutoff_values = super().find_convergence_values(relative_values, cutoff_values, conv_criterion)

        if y_log: 
            ax.semilogy(cutoff_values_new, relative_values, label=self.label_relative_values,**plot_settings['curve_settings'])
        else:
            ax.plot(cutoff_values_new, relative_values, label=self.label_relative_values,**plot_settings['curve_settings'])
        
        ax.axhline(
            y=conv_criterion,
            label=self.label_conv_criterion,
            **plot_settings['criterion_settings']
        )
        
        if show_fill:
            ax.axhspan(
                ymin=0,
                ymax=conv_criterion,
                **plot_settings["fill_settings"]
            )
        
        ax.set_title(self.title_name, fontdict=plot_settings['fontdict_title'])
        ax.set_ylabel(self.axis_y_name)
        ax.set_xlabel(self.axis_x_name)
        ax.legend(loc=plot_settings["legend_loc"])

        return fig
    
    @staticmethod
    def get_cutoff_convergence_plot(energies: np.ndarray,
                                    cutoff_values: np.ndarray,
                                    conv_criterion: float,
                                    title_name: str = "Total energy vs Energy cutoff",
                                    axis_x_name: str = "Energy cutoff [eV]",
                                    axis_y_name: str = r'$|\,\Delta E \,| $', #r'$|\Delta E|$ [meV/#atoms]', 
                                    label_relative_values: str = r'$|\,\Delta E \,| $',
                                    SI_unit: str = "eV",
                                    use_SI_prefixes:str = "milli",
                                    show_fill: bool = True,
                                    y_log: bool = True,
                                    **settings):
        """
        Create and return a convergence plot of energies against cutoff values.

        Parameters:
        ----------
        energies : np.ndarray
            Array of energy values to analyze.
        cutoff_values : np.ndarray
            Array of cutoff values corresponding to energies.
        conv_criterion : float
            Convergence threshold value.
        title_name : str
            Title for the plot.
        axis_x_name : str
            Label for the x-axis.
        axis_y_name : str
            Label for the y-axis.
        label_relative_values : str
            Label for the relative energies in the plot.
        label_conv_criterion_unit : str
            Unit label for the convergence criterion.
        show_fill : bool
            Whether to show the fill area under the convergence line (default is True).
        y_log : bool
            Whether to use a logarithmic scale for the y-axis (default is True).
        settings : dict
            Additional plot settings.
        
        Returns:
        -------
        fig : matplotlib.figure.Figure
            The figure object created.
        """

        conver_unit_1 = SIUnitConverter(value=energies,unit=SI_unit)
        new_energy_values, new_unit = conver_unit_1.convert(prefix=use_SI_prefixes)

        conver_unit_2 = SIUnitConverter(value=conv_criterion,unit=SI_unit)
        new_conv_criterion,new_unit_conv_criterion = conver_unit_2.convert(prefix=use_SI_prefixes)

        
        plot = ConvergencePlot(
            title_name=title_name,
            axis_x_name=axis_x_name,
            axis_y_name=f'{axis_y_name} [{new_unit}]',
            label_relative_values= f'{label_relative_values}',
            label_conv_criterion=f'Criterion {new_conv_criterion} [{new_unit_conv_criterion}]'
        )
        
        relative_energies = plot.create_abs_diff_values(new_energy_values)

        fig = plot.plot_convergence(
                                    relative_values=relative_energies,
                                    cutoff_values=cutoff_values,
                                    conv_criterion=new_conv_criterion,
                                    show_fill=show_fill,
                                    y_log=y_log,
                                    **settings
                                    )
        
        return fig
    
    @staticmethod
    def get_kpoint_convergence_plot(energies: np.ndarray,
                                    cutoff_values: np.ndarray,
                                    conv_criterion: float,
                                    title_name: str = "Total energy vs K-Points density",
                                    axis_x_name: str = "K-Points density",
                                    axis_y_name: str = r'$|\,\Delta E \,| $', #r'$|\Delta E|$ [meV/#atoms]', 
                                    label_relative_values: str = r'$|\,\Delta E \,| $',
                                    SI_unit: str = "eV",
                                    use_SI_prefixes:str = "milli",
                                    show_fill: bool = True,
                                    y_log: bool = True,
                                    **settings):
        """
        Create and return a convergence plot of energies against cutoff values.

        Parameters:
        ----------
        energies : np.ndarray
            Array of energy values to analyze.
        cutoff_values : np.ndarray
            Array of cutoff values corresponding to energies.
        conv_criterion : float
            Convergence threshold value.
        title_name : str
            Title for the plot.
        axis_x_name : str
            Label for the x-axis.
        axis_y_name : str
            Label for the y-axis.
        label_relative_values : str
            Label for the relative energies in the plot.
        label_conv_criterion_unit : str
            Unit label for the convergence criterion.
        show_fill : bool
            Whether to show the fill area under the convergence line (default is True).
        y_log : bool
            Whether to use a logarithmic scale for the y-axis (default is True).
        settings : dict
            Additional plot settings.
        
        Returns:
        -------
        fig : matplotlib.figure.Figure
            The figure object created.
        """

        conver_unit_1 = SIUnitConverter(value=energies,unit=SI_unit)
        new_energy_values, new_unit = conver_unit_1.convert(prefix=use_SI_prefixes)

        conver_unit_2 = SIUnitConverter(value=conv_criterion,unit=SI_unit)
        new_conv_criterion,new_unit_conv_criterion = conver_unit_2.convert(prefix=use_SI_prefixes)

        
        plot = ConvergencePlot(
            title_name=title_name,
            axis_x_name=axis_x_name,
            axis_y_name=f'{axis_y_name} [{new_unit}]',
            label_relative_values=f'{label_relative_values}',
            label_conv_criterion=f'Criterion {new_conv_criterion} [{new_unit_conv_criterion}]'
        )
        
        relative_energies = plot.create_abs_diff_values(new_energy_values)

        fig = plot.plot_convergence(
                                    relative_values=relative_energies,
                                    cutoff_values=cutoff_values,
                                    conv_criterion=new_conv_criterion,
                                    show_fill=show_fill,
                                    y_log=y_log,
                                    **settings
                                    )
        
        return fig