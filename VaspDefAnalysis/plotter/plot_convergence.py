import sys
import numpy as np
import matplotlib.pyplot as plt
import os

# Add path to my package (ensure you add the directory, not the file itself)
#sys.path.append(os.path.abspath('VaspDefAnalysis/convergence'))

from VaspDefAnalysis.utils.convergence import ConvergenceTools
from VaspDefAnalysis.utils.tool_pool import SIUnitConverter

class ConvergencePlot(ConvergenceTools):
    """
    The ConvergencePlot class extends the ConvergenceTools to provide functionalities 
    for visualizing convergence tests, particularly for plotting energy convergence with respect to 
    cutoff and K-Point density parameters. 

    The class utilizes matplotlib for plotting and supports customization of the plot's appearance 
    through various settings. Users can generate and customize plots of energy differences (relative relative_values) 
    to visualize how calculated properties converge with different computational parameters.
    """
    def __init__(self, 
                 title_name: str,
                 axis_x_name: str,
                 axis_y_name: str):
        """Initialize the plot with labels and axis names."""
        self.title_name = title_name
        self.axis_x_name = axis_x_name
        self.axis_y_name = axis_y_name

    def plot_setting(self, **plot_setting):
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
            "label_size": 10,
            "x_y_label_size": 16,
            "figsize": (8, 6),
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
        dict_keys = ['fontdict_title','curve_settings','criterion_settings','fill_settings']
        for dict_key in dict_keys:
            if dict_key in plot_setting:
                plot_setting[dict_key] = {**default_settings[dict_key], **plot_setting[dict_key]}
    
        validated_settings = {**default_settings, **plot_setting}
        return validated_settings
    
    def plot_diff_convergence(self,
                         relative_values: np.ndarray,
                         cutoff_values: np.ndarray,
                         conv_criterion: float,
                         label_relative_values:str,
                         label_conv_criterion:str,
                         show_fill: bool = True,
                         y_log: bool = True,
                         **settings):
        """
        Plot the convergence data with relative relative_values vs cut-off values.

        Parameters:
        ----------
        relative_values : np.ndarray
            Relative relative_values for plotting.
        cutoff_values : np.ndarray
            The cut-off values for plotting.
        conv_criterion : float
            The convergence threshold.
        label_relative_values:
            The label of the relative values
        label_conv_criterion
            The label of the ceonvergence criterion
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
        self.label_relative_values = label_relative_values
        self.label_conv_criterion = label_conv_criterion

        
        #if len(relative_values) != len(cutoff_values):
        #    raise ValueError("Lengths of relative_values and cutoff_values must match.")

        plot_settings = self.plot_setting(**settings)
        
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
                ymin=0.0,
                ymax=conv_criterion,
                **plot_settings["fill_settings"]
            )
        
        ax.set_title(self.title_name, fontdict=plot_settings['fontdict_title'])
        ax.set_ylabel(self.axis_y_name,size=plot_settings["x_y_label_size"])
        ax.set_xlabel(self.axis_x_name,size=plot_settings["x_y_label_size"])
        ax.legend(loc=plot_settings["legend_loc"],fontsize = plot_settings['label_size'])

        # Adjust layout to prevent clipping of labels
        fig.tight_layout()

        return fig
        
    def plot_convergence(self,
                         relative_values: np.ndarray,
                         cutoff_values: np.ndarray,
                         label_relative_values:str = None,
                         y_log: bool = False,
                         **settings):
        """
        Plot the convergence data with relative relative_values vs cut-off values.

        Parameters:
        ----------
        relative_values : np.ndarray
            Relative relative_values for plotting.
        cutoff_values : np.ndarray
            The cut-off values for plotting.
        settings : dict
            Additional plot settings.
        
        Returns:
        -------
        fig : matplotlib.figure.Figure
            The figure object created.
        """
        
        if len(relative_values) != len(cutoff_values):
            raise ValueError("Lengths of relative_values and cutoff_values must match.")

        plot_settings = self.plot_setting(**settings)
        
        fig, ax = plt.subplots(figsize=plot_settings["figsize"])
       

        if y_log: 
            ax.semilogy(cutoff_values, relative_values, label=label_relative_values,**plot_settings['curve_settings'])
        else:
            ax.plot(cutoff_values, relative_values, label=label_relative_values,**plot_settings['curve_settings'])
        
        ax.set_title(self.title_name, fontdict=plot_settings['fontdict_title'])
        ax.set_ylabel(self.axis_y_name,size=plot_settings["x_y_label_size"])
        ax.set_xlabel(self.axis_x_name,size=plot_settings["x_y_label_size"])
        if  label_relative_values != None :
            ax.legend(loc=plot_settings["legend_loc"],fontsize = plot_settings['label_size'])

        # Adjust layout to prevent clipping of labels
        fig.tight_layout()

        return fig
    
    @staticmethod
    def get_diff_convergence_plot(relative_values: np.ndarray,
                                  cutoff_values: np.ndarray,
                                  conv_criterion: float, # [eV]
                                  title_name: str = "Relative energy vs Energy cutoff",
                                  axis_x_name: str = "Energy cutoff [eV]",
                                  axis_y_name: str = r'$|\,\Delta E \,| $', #r'$|\Delta E|$ [meV/#atoms]', 
                                  label_relative_values: str = r'$|\,\Delta E \,| $',
                                  SI_unit: str = "eV",
                                  use_SI_prefixes:str = "milli",
                                  show_fill: bool = True,
                                  y_log: bool = True,
                                  **settings):
        """
        Create and return a convergence plot of relative_values against cutoff values.

        Parameters:
        ----------
        relative_values : np.ndarray
            Array of energy values to analyze.
        cutoff_values : np.ndarray
            Array of cutoff values corresponding to relative_values.
        conv_criterion : float
            Convergence threshold value.
        title_name : str
            Title for the plot.
        axis_x_name : str
            Label for the x-axis.
        axis_y_name : str
            Label for the y-axis.
        label_relative_values : str
            Label for the relative relative_values in the plot.
        label_conv_criterion_unit : str
            Unit label for the convergence criterion.
        SI_unit: str = "eV".
            Specifies the unit of energy, defaulting to electronvolts (eV).
        use_SI_prefixes: str = "milli",  
            Specifies the SI prefix to use for conversion, defaulting to milli (e.g., converting eV to meV).
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

        conver_unit_1 = SIUnitConverter(value=relative_values,unit=SI_unit)
        new_energy_values, new_unit = conver_unit_1.convert(prefix=use_SI_prefixes)

        conver_unit_2 = SIUnitConverter(value=conv_criterion,unit=SI_unit)
        new_conv_criterion,new_unit_conv_criterion = conver_unit_2.convert(prefix=use_SI_prefixes)

        #new_conv_criterion_round = round(new_conv_criterion,3)
        
        plot = ConvergencePlot(
            title_name=title_name,
            axis_x_name=axis_x_name,
            axis_y_name=f'{axis_y_name} [{new_unit}]')
        
        relative_relative_values = plot.create_abs_diff_values(new_energy_values)

        fig = plot.plot_diff_convergence(
                                    relative_values=relative_relative_values,
                                    cutoff_values=cutoff_values,
                                    conv_criterion=new_conv_criterion,
                                    label_relative_values= f'{label_relative_values}',
                                    label_conv_criterion=f'Criterion {new_conv_criterion} [{new_unit_conv_criterion}]',
                                    show_fill=show_fill,
                                    y_log=y_log,
                                    **settings
                                    )
        
        return fig
    
    @staticmethod
    def get_convergence_plot(relative_values: np.ndarray,
                             cutoff_values: np.ndarray,
                             title_name: str = "Relative energy vs Energy cutoff",
                             axis_x_name: str = "Energy cutoff [eV]",
                             axis_y_name: str = 'Relative relative_values', #r'$|\Delta E|$ [meV/#atoms]', 
                             label_relative_values: str = None,
                             SI_unit: str = "eV",
                             use_SI_prefixes:str = "milli",
                             y_log: bool = False,
                             **settings):
        """
        Create and return a convergence plot of relative_values against cutoff values.

        Parameters:
        ----------
        relative_values : np.ndarray
            Array of energy values to analyze.
        cutoff_values : np.ndarray
            Array of cutoff values corresponding to relative_values.
        title_name : str
            Title for the plot.
        axis_x_name : str
            Label for the x-axis.
        axis_y_name : str
            Label for the y-axis.
        label_relative_values : str
            Label for the relative relative_values in the plot.
        label_conv_criterion_unit : str
            Unit label for the convergence criterion.
        SI_unit: str = "eV".
            Specifies the unit of energy, defaulting to electronvolts (eV).
        use_SI_prefixes: str = "milli",  
            Specifies the SI prefix to use for conversion, defaulting to milli (e.g., converting eV to meV).
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

        conver_unit_1 = SIUnitConverter(value=relative_values,unit=SI_unit)
        new_energy_values, new_unit = conver_unit_1.convert(prefix=use_SI_prefixes)

        plot = ConvergencePlot(
            title_name=title_name,
            axis_x_name=axis_x_name,
            axis_y_name=f'{axis_y_name} [{new_unit}]')
        
        fig = plot.plot_convergence(
                                    relative_values=new_energy_values,
                                    cutoff_values=cutoff_values,
                                    label_relative_values= label_relative_values,
                                    y_log=y_log,
                                    **settings
                                    )
        
        return fig