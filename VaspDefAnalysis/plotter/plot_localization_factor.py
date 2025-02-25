# Packages
import numpy as np
import matplotlib.pyplot as plt

from VaspDefAnalysis.defect.defect_analisys import DefectAnalisys
from VaspDefAnalysis.read_vasp.vasprun_analysis import VaspRunAnalysis
#from VaspDefAnalysis.defect.localization_factor import get_ion_orbital_total_weight_neighbors_to_defects,get_ipr_values
from VaspDefAnalysis.defect.localization_factor import LocalizedStates
from VaspDefAnalysis.plotter.tool_plotter import generate_fraction_labels_for_kpoints

class PlotLocalizedStates:
    """
    Plot Kohn-Sham localization eigenvalue using output of VASP, It can handle two methods (Vienna Ab initio Simulation Package): 
    using the ion orbital weights (IOWs) and Inverse Participation Ratio (IPR).
    """
    def __init__(self,
                 kpoints_dict: dict, 
                 eigenvalues_dict: dict,
                 localized_paramter_dict: dict,
                 ):
        """
        Initialize the PlotLocalizedStates class with data dictionaries.

        Parameters:
        ----------
        kpoints_dict : dict
            Dictionary mapping spins to k-point fractional coordinates.
        eigenvalues_dict : dict
            Dictionary containing eigenvalues for each spin and k-point.
        localized_paramter_dict : dict
            Dictionary containing localization factors for each spin and k-point.
        """
        self.kpoints_dict = kpoints_dict
        self.eigenvalues_dict = eigenvalues_dict
        self.localized_paramter_dict = localized_paramter_dict

    def generate_nice_x_labels(self, kpt_coords, line_break="\n"):
        """
        Generate x-axis labels for k-points based on their fractional coordinates.

        Parameters:
        ----------
        kpt_coords : list
            Fractional coordinates of k-points.
        line_break : str, optional
            Character to separate coordinates in the label. Default is '\n'.

        Returns:
        -------
        list
            Formatted labels for k-points.
        """
        nice_x_labels = generate_fraction_labels_for_kpoints(kpt_coords=kpt_coords, line_break=line_break)
        return nice_x_labels
    
    def plot_setting(self, **plot_settings):
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
            "scatter_settings": {'s': 100, 'linewidths': 0.01, 'edgecolor': 'black', "cmap": "viridis"},
            "vbm_color": "black",  
            "cbm_color": "black",     
            "vbm_line_style": "--",
            "cbm_line_style": "--",
            "show_vbm_cbm":True,
            "fill_up_color_vb": "grey",
            "fill_up_color_cb": "grey", 
            "fill_up_alpha": 0.3,
            "title_names": {"up": "Kohn-Sham Eigenvalues spin-up", "down": "Kohn-Sham Eigenvalues spin-down"},
            "fontdict_title": {"family": "serif", "color": "black", "weight": "bold", "size": 14},
            "xlabel": "K-Points",
            "ylabel": "Eigenvalues [eV]",
            "colorbar_label": "Location factor",
            "label_size": 12,
            "figsize": 6,
            "layout": "vertical",
            "IPR":{"gvec": None, "Cg":None, "ngrid":None,"rescale":None, "kr_phase":False, "r0":[0.0, 0.0, 0.0]}
        }
        
        # Validate keys
        valid_keys = default_settings.keys()
        invalid_keys = [key for key in plot_settings if key not in valid_keys]
        if invalid_keys:
            raise ValueError(f"Invalid keys in plot_setting: {invalid_keys}")
    
        # Update nested dictionary-based settings with user-provided values
        dict_keys = ['fontdict_title', 'scatter_settings', 'title_names']
        for dict_key in dict_keys:
            if dict_key in plot_settings:
                plot_settings[dict_key] = {**default_settings[dict_key], **plot_settings[dict_key]}
    
        # Update default settings with user-provided settings
        validated_settings = {**default_settings, **plot_settings}
        return validated_settings
    
    def plot_localized_state(
        self,
        VBM: float,
        CBM: float,
        y_limit: tuple | None ="(VBM-1.5,CBM+1.5)",
        fermi_energy_reference: bool = True,
        show_fill_up: bool = True,
        **plot_settings
        ) -> plt.Figure:
        """
        Plot localized states for spins with customization.

        Parameters:
        ----------
        VBM : float or None
            Valence Band Maximum energy level.
        CBM : float or None
            Conduction Band Minimum energy level.
        y_limit : tuple or None, optional
            Limits for the y-axis. Default is calculated based on VBM and CBM.
        fermi_energy_reference : bool, optional
            Whether to use VBM as a reference energy. Default is True.
        show_fill_up : bool, optional
            Whether to fill regions above CBM and below VBM. Default is True.
        plot_setting : dict
            Custom settings for plot appearance.

        Returns:
        -------
        plt.Figure
            The generated figure object.
        """
        # Handle plot settings
        plot_settings = self.plot_setting(**plot_settings)

        # Determine layout and create subplots
        num_spins = len(self.eigenvalues_dict)
        layout = plot_settings["layout"]
        rows, cols = (1, num_spins) if layout == "horizontal" else (num_spins, 1)
        fig, axes = plt.subplots(rows, cols, figsize=(plot_settings["figsize"] * cols, plot_settings["figsize"] * rows))
        
        # Ensure axes is always iterable
        axes = [axes] if num_spins == 1 else axes
        
        spin_idx = 0
        for (spin_key, kpoint_keys), (spin_key_, kpoint_keys_) in zip(self.eigenvalues_dict.items(), self.localized_paramter_dict.items()):
            ax = axes[spin_idx]

            # Generate x-axis labels
            kpt_coords = [self.kpoints_dict[spin_key][kpoint_key] for kpoint_key in kpoint_keys]
            _x_labels = self.generate_nice_x_labels(kpt_coords)
            x_values = list(range(len(kpt_coords)))

            # Save dat for the different kpoints for scatter plot
            min_KS, max_KS = [], []
            x_values_list, eigenvalues_list, localized_values_list = [], [], []

            kpoint_idx = 0 
            for (kpoint_key, eigenvalues), (kpoint_key_, localized_values) in zip(kpoint_keys.items(), kpoint_keys_.items()):
                if fermi_energy_reference:
                    eigenvalues = list(np.array(eigenvalues)- VBM)
                eigenvalues_list.extend(eigenvalues)
                x_values_list.extend([x_values[kpoint_idx]] * len(eigenvalues))
                localized_values_list.extend(localized_values)
                min_KS.append(min(eigenvalues))
                max_KS.append(max(eigenvalues))
                kpoint_idx += 1 

            # Scatter plot
            scatter = ax.scatter(x_values_list, eigenvalues_list, c=localized_values_list, **plot_settings["scatter_settings"])  
            cbar = plt.colorbar(scatter, ax=ax, label=plot_settings["colorbar_label"])
            cbar.set_label(label=plot_settings["colorbar_label"], fontsize=plot_settings["label_size"])

            # Set axis labels and titles
            ax.set_xticks(x_values)
            ax.set_xticklabels(_x_labels, rotation=0.0, ha="right")
            ax.set_xlabel(plot_settings["xlabel"], size=plot_settings["label_size"])
            ax.set_ylabel(plot_settings["ylabel"], size=plot_settings["label_size"])
            title = plot_settings["title_names"].get("up" if spin_key == 'spin 1' else "down")
            ax.set_title(title, fontdict=plot_settings["fontdict_title"])

            # If the energies are referenced to energy fermi
            y_value_VBM = VBM - VBM if fermi_energy_reference else VBM
            y_value_CBM = CBM - VBM if fermi_energy_reference else CBM
            
            # Default y-limit
            if y_limit == "(VBM-1.5,CBM+1.5)":
                y_limit = (y_value_VBM - 1.5, y_value_CBM + 1.5)
                ax.set_ylim(y_limit)
            else: 
                ax.set_ylim(y_limit)

            # Plot VBM/CBM lines and fill regions
            if show_fill_up:
                ax.axhline(y=y_value_VBM, color=plot_settings["vbm_color"], linestyle=plot_settings["vbm_line_style"])
                ax.axhline(y=y_value_CBM, color=plot_settings["cbm_color"], linestyle=plot_settings["cbm_line_style"])
            if show_fill_up:
                ax.axhspan(min(min_KS), y_value_VBM, color=plot_settings["fill_up_color_vb"], alpha=plot_settings["fill_up_alpha"])
                ax.axhspan(y_value_CBM, max(max_KS), color=plot_settings["fill_up_color_cb"], alpha=plot_settings["fill_up_alpha"])

            spin_idx += 1

        # Adjust subplot spacing and return the figure
        fig.tight_layout() 
        #plt.close()
        return fig
    
    @staticmethod
    def get_plot_LS_using_IOWs(
                               perfect_structure_path:str,   
                               defect_structure_path:str,
                               defect_vasprun_path:str,
                               VBM: float,
                               CBM: float,
                               y_limit: tuple | None ="(VBM-1.5,CBM+1.5)",
                               fermi_energy_reference: bool = True,
                               show_fill_up: bool = True,
                               tolerance=1e-1,
                               add_neighbors_up:int=1,
                               norm:bool=True,
                               **plot_settings
                                ) -> plt.Figure:
        """
        get_plot_LS_using_IOWs to plot Kohn-Sham eigenvalues for each k-point but also include the "localization factor" using the ion orbital weights (IOWs).

        Parameters:
        ----------
        perfect_structure_path : str
            Path to the file containing the perfect (non-defective) structure.
        defect_structure_path : str
            Path to the file containing the defective structure.
        vasprun_defect_path : str
            Path to the `vasprun.xml` file for the defect calculation.
        VBM : float
            The Valence Band Maximum (VBM) energy in eV.
        CBM : float
            The Conduction Band Minimum (CBM) energy in eV.
        y_limit : tuple | None, optional
            Tuple defining the y-axis range for the plot; default is calculated as `(VBM-1.5, CBM+1.5)`.
        fermi_energy_reference : bool, optional
            Whether to use the Fermi energy as a reference for eigenvalue alignment. Default is `True`.
        show_fill_up : bool, optional
            If `True`, fill the occupied states up to the Fermi level. Default is `True`.
        tolerance : float, optional
            Tolerance for identifying neighboring ions around the defect site. Default is `1e-1`.
        add_neighbors_up : int, optional
            Number of additional neighbors to include around the defect site. Default is `1`.
        norm : bool, optional
            If True, normalizes the weights by the total weight for each k-point. Defaults to `True`.
        **plot_setting
            Additional keyword arguments for customizing the plot appearance.

        Returns:
        -------
        plt.Figure
            A matplotlib figure object showing the localized state plot.
        """

        # Intance VaspRunAnalysis class
        vasprun_analysis = VaspRunAnalysis(vasprun_path=defect_vasprun_path)

        # Extract Kohn-Sham eigenvalues
        Kohn_Sham_eigenvalues_dic,Kohn_Sham_occupancy_state = vasprun_analysis.get_Kohn_Sham_eigenvalues_and_occupancy()

        # Get k-point values
        kpoint_values_dic = vasprun_analysis.get_kpoint_values()


        # Calculate ion orbital total weights for neighbors around defects
        localization_factor_dic_using_IOWs =LocalizedStates.get_localization_factor_using_IOWs(perfect_structure_path=perfect_structure_path,
                                                                                               defect_structure_path=defect_structure_path,
                                                                                               defect_vasprun_path=defect_vasprun_path,
                                                                                               tolerance=tolerance,
                                                                                               add_neighbors_up=add_neighbors_up,
                                                                                               norm=norm)

        # Create an instance of PlotLocalizedStates class 
        plotter = PlotLocalizedStates(kpoints_dict=kpoint_values_dic,eigenvalues_dict=Kohn_Sham_eigenvalues_dic,localized_paramter_dict=localization_factor_dic_using_IOWs)
        fig = plotter.plot_localized_state(VBM=VBM,CBM=CBM,y_limit=y_limit,fermi_energy_reference=fermi_energy_reference,show_fill_up=show_fill_up,colorbar_label="IOWs",**plot_settings)
        return fig
    
    @staticmethod
    def get_plot_LS_using_IPR(WAVECAR_path:str,
                              vasprun_path:str,
                              VBM: float,
                              CBM: float,
                              y_limit: tuple | None ="(VBM-1.5,CBM+1.5)",
                              fermi_energy_reference: bool = True,
                              show_fill_up: bool = True,
                              lsorbit:bool=False, 
                              lgamma:bool=False,
                              gamma_half:str='x',
                              omp_num_threads:int=1,
                              norm:bool=True,
                              **plot_setting
                              ):
        
        """
        Function to plot Kohn-Sham eigenvalues with the Inverse Participation Ratio (IPR) as a localization measure.
    
        Parameters:
        ----------
        WAVECAR_path : str
            Path to the WAVECAR file containing the wavefunction data.
        vasprun_path : str
            Path to the `vasprun.xml` file for the VASP run.
        VBM : float
            The Valence Band Maximum (VBM) energy in eV.
        CBM : float
            The Conduction Band Minimum (CBM) energy in eV.
        y_limit : tuple | None, optional
            Tuple defining the y-axis range for the plot (default is `(VBM-1.5, CBM+1.5)`).
        fermi_energy_reference : bool, optional
            Whether to use the Fermi energy as a reference for eigenvalue alignment (default is `True`).
        show_fill_up : bool, optional
            If `True`, fill the occupied states up to the Fermi level (default is `True`).
        lsortbit : bool, optional
            If `True`, include localized state calculation using IPR (default is `False`).
        lgamma : bool, optional
            If `True`, include Gamma points for the calculation (default is `False`).
        gamma_half : str, optional
            Specify the type of gamma scaling for the calculation (default is `'x'`).
        omp_num_threads : int, optional
            The number of threads to use for parallel processing (default is `1`).
        norm : normalized Cg?
            It is by default  `True`

        **plot_setting:
            Additional keyword arguments for customizing the plot appearance and IPR consideration.
        
        Returns:
        -------
        plt.Figure
            A matplotlib figure object showing the localized state plot.
        
        """

        r""" 
        The normalizatition is defined:
        
        \sum_{ijk} | \phi_{ijk} | ^ 2 = 1
        """
        
        # Intance VaspRunAnalysis class
        vasprun_analysis = VaspRunAnalysis(vasprun_path=vasprun_path)

        # Extract Kohn-Sham eigenvalues
        Kohn_Sham_eigenvalues_dic,Kohn_Sham_occupancy_state = vasprun_analysis.get_Kohn_Sham_eigenvalues_and_occupancy()
        
        # Get k-point values
        kpoint_values_dic = vasprun_analysis.get_kpoint_values()

        ipr_values_dic = LocalizedStates.get_localization_factor_using_IPR(WAVECAR_path=WAVECAR_path,
                                                                           lsorbit=lsorbit,
                                                                           lgamma=lgamma,
                                                                           gamma_half=gamma_half,
                                                                           omp_num_threads=omp_num_threads,
                                                                           norm=norm,**plot_setting)
        
        # Create an instance of PlotLocalizedStates class 
        plotter = PlotLocalizedStates(kpoints_dict=kpoint_values_dic,eigenvalues_dict=Kohn_Sham_eigenvalues_dic,localized_paramter_dict=ipr_values_dic)
        fig = plotter.plot_localized_state(VBM=VBM,CBM=CBM,y_limit=y_limit,fermi_energy_reference=fermi_energy_reference,show_fill_up=show_fill_up,colorbar_label="IPRs",**plot_setting)
        return fig
        




