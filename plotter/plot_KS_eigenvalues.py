import matplotlib.pyplot as plt
import numpy as np

from VaspDefAnalysis.read_vasprun.vasprun_analysis import VaspRunAnalysis
from VaspDefAnalysis.plotter.tool_plotter import classify_eigenvalues,generate_fraction_labels_for_kpoints

class PlotKohnShamEigenvalue:
    def __init__(self,
                eigenvalues_dict:dict, 
                occupancy_dict:dict, 
                kpoints_dict:dict
                ):
        
        """
        Initialize the PlotKohnShamEigenvalue class with eigenvalues, occupancies, and k-point data.

        Parameters
        ----------
        eigenvalues_dict : dict
            A dictionary containing eigenvalues for different spins and k-points. 
            Structure: {spin_key: {kpoint_key: eigenvalue_list}}.

        occupancy_dict : dict
            A dictionary containing occupancies for different spins and k-points. 
            Structure: {spin_key: {kpoint_key: occupancy_list}}.

        kpoints_dict : dict
            A dictionary mapping k-point identifiers to their coordinates.
            Structure: {spin_key: {kpoint_key: [kx, ky, kz]}}.

        Attributes
        ----------
        self.eigenvalues_dict : dict
            Stores the eigenvalues for the instance.

        self.occupancy_dict : dict
            Stores the occupancies for the instance.

        self.kpoints_dict : dict
            Stores the k-point coordinates for the instance.

        Notes
        -----
        These dictionaries should be consistent in structure across spins and k-points.
        """
        
        self.eigenvalues_dict = eigenvalues_dict
        self.occupancy_dict = occupancy_dict
        self.kpoints_dict = kpoints_dict

    def classify_eigenvalues_to_occupancy(self)-> dict:
        
        """
        Classify eigenvalues based on their occupancy into three categories: 
        occupied, unoccupied, and partially occupied.

        Returns
        -------
        dict
            A nested dictionary where eigenvalues are categorized for each spin and k-point:
            - Keys at the first level are spin identifiers (e.g., "spin 1", "spin 2").
            - Keys at the second level are k-point identifiers.
            - Values are dictionaries with three lists: "occupied", "unoccupied", and "partial".
              Each list contains eigenvalues corresponding to the respective category.

        Classification Rules
        --------------------
        - Occupied: Occupancy >= 0.9
        - Unoccupied: Occupancy <= 0.1
        - Partial: 0.1 < Occupancy < 0.9

        Notes
        -----
        - This function assumes that `self.eigenvalues_dict` contains the eigenvalues for each spin and k-point.
        - It also assumes that `self.occupancy_dict` provides the corresponding occupancy values.
        - If a k-point has no occupancy information in `self.occupancy_dict`, an empty list is used by default.

        Example Output
        --------------
        {
            "spin 1": {
                "k-point 1": {
                    "occupied": [...],      # List of occupied eigenvalues
                    "unoccupied": [...],    # List of unoccupied eigenvalues
                    "partial": [...]        # List of partially occupied eigenvalues
                },
                ...
            },
            ...
        }
        """
        classified_eigenvalues = classify_eigenvalues(eigenvalues_dict=self.eigenvalues_dict,occupancy_dict=self.occupancy_dict)

        return classified_eigenvalues

    def generate_nice_x_labels(self, kpt_coords, line_break="\n"):
        """
        Generate x-axis labels for k-points based on their fractional coordinates.
        """
        nice_x_labels = generate_fraction_labels_for_kpoints(kpt_coords=kpt_coords,line_break=line_break)
        return nice_x_labels
    
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
        "occupied_color": "red",
        "occupied_marker": "o",
        "unoccupied_color": "blue",
        "unoccupied_marker": "o",
        "partial_color": "green",
        "partial_marker": "o",
        "scatter_settings":{'s': 100,'linewidths': 0.01,'edgecolor': 'black'},
        "vbm_color": "black",  
        "cbm_color": "black",     
        "vbm_line_style": "--",
        "cbm_line_style": "--",
        "show_vbm_cbm":True,
        "fill_up_color_vb": "grey",
        "fill_up_color_cb": "grey", 
        "fill_up_alpha": 0.3,
        "title_names":{"up":"Kohn-Sham Eigenvalues spin-up","down":"Kohn-Sham Eigenvalues spin-down"},
        "fontdict_title": {"family": "serif","color": "black","weight": "bold","size": 14},
        "xlabel":"K-Points",
        "ylabel":"Eigenvalues [eV]",
        "label_size": 12,
        "legend_loc": "upper right",
        "figsize": 6,   # horizontal (6,6*2), vertical (6*2,6)
        "layout": "vertical"
        }
        
        # Validate keys
        valid_keys = default_settings.keys()
        invalid_keys = [key for key in plot_setting if key not in valid_keys]
        if invalid_keys:
            raise ValueError(f"Invalid keys in plot_setting: {invalid_keys}")
    
        # Separate out dictionary-based settings to be updated
        dict_keys = ['fontdict_title', 'scatter_settings', 'title_names']
        for dict_key in dict_keys:
            if dict_key in plot_setting:
                # Update the existing dictionary with new settings from plot_setting
                plot_setting[dict_key] = {**default_settings[dict_key], **plot_setting[dict_key]}
    
        # Update default settings with user-provided settings
        validated_settings = {**default_settings, **plot_setting}
        return validated_settings

    def plot_KS_eigenvalues(self, 
                         VBM:float,
                         CBM:float,
                         y_limit:tuple|None="(VBM-1.5,CBM+1.5)",
                         fermi_energy_reference:bool = True,
                         show_fill_up:bool=True,
                         **plot_setting
                         )-> plt.Figure:
        
        """
        Plot Kohn-Sham (KS) eigenvalues

        Parameters
        ----------
        VBM : float
            Valence Band Maximum (VBM) energy level, used as a reference for plotting band edges.
        CBM : float
            Conduction Band Minimum (CBM) energy level, used as a reference for plotting band edges.
        y_limit : tuple, optional
            A tuple specifying the y-axis limits for the plot (e.g., (y_min, y_max)).
            Default is None, meaning no specific limits are set.
        fermi_energy_reference : bool, optional
            If True, plot eigenvalues relative to the Fermi energy; otherwise, use absolute eigenvalues.
            Default is True.
        show_fill_up : bool, optional
            If True, fills the regions below VBM and above CBM to visually represent occupied/unoccupied states.
            Default is False.
        **plot_setting : dict
            Additional plot customization settings, such as:
                - occupied_color: Color for occupied eigenvalues.
                - occupied_marker: Marker style for occupied eigenvalues.
                - unoccupied_color: Color for unoccupied eigenvalues.
                - unoccupied_marker: Marker style for unoccupied eigenvalues.
                - partial_color: Color for partially occupied eigenvalues.
                - partial_marker: Marker style for partially occupied eigenvalues.
                - scatter_settings: settings scatter plot
                - vbm_color, cbm_color: Colors for VBM and CBM lines.
                - vbm_line_style, cbm_line_style: Line styles for VBM and CBM.
                - fill_up_color_vb, fill_up_color_cb: Colors for valence and conduction band fill.
                - fill_up_alpha: Opacity for shaded regions.
                - title_names: Titles for spin plots, provided as a dictionary with keys 'up' and 'down'.
                - fontdict_title: Font settings for subplot titles.
                - xlabel, ylabel: Labels for x and y axes.
                - label_size: Font size for axis labels.
                - legend_loc: Location for the legend.

        Returns
        -------
        None
            Generates and displays a plot of KS eigenvalues. No return value.
        """
        
        # Handle plot settings
        plot_settings = self.plot_setting_kwargs(**plot_setting)
                
        # Use classified eigenvalues for better organization
        classified_eigenvalues = self.classify_eigenvalues_to_occupancy()
        
        # Number of subplots needed (one for each spin)
        num_spins = len(classified_eigenvalues)  
        
        # Determine the number of rows and columns based on layout
        if plot_settings["layout"] == "horizontal":
            rows, cols = 1, num_spins
        elif plot_settings["layout"] == "vertical":
            rows, cols = num_spins, 1
        else:
            raise ValueError("Invalid layout. Choose 'horizontal' or 'vertical'.")
        
        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(plot_settings["figsize"] * cols,plot_settings["figsize"] * rows))
        
        # If only one subplot, axes will not be an array, so handle it accordingly
        if num_spins == 1:
            axes = [axes]

        for spin_idx, (spin_key, kpoint_keys) in enumerate(classified_eigenvalues.items()):

            # Get the corresponding subplot for this spin
            ax = axes[spin_idx] 

            # Extract k-point coordinates for x-axis labels
            kpt_coords = [self.kpoints_dict[spin_key][kpoint_key] for kpoint_key in kpoint_keys]
            _x_labels = self.generate_nice_x_labels(kpt_coords)

            # Create a list of x-values corresponding to each k-point for the plot
            x_values = list(range(len(kpoint_keys)))

            # Save the minimum occupied  and maximum unoccupied eigenvalues for different kpoints (future references)
            minimum_ocucupied_eigenvalues = []
            maximum_unoccupied_eigenvalues = []
            for kpoint_idx, (kpoint_key, bands) in enumerate(kpoint_keys.items()):
                
                # Fermi energy reference 
                if fermi_energy_reference:
                    occupied_eigenvalues = np.array(bands["occupied"]) - VBM
                    unoccupied_eigenvalues = np.array(bands["unoccupied"]) - VBM
                    partial_occupied_eigenvalue = np.array(bands["partial"]) - VBM
                    # Minimum occupied  and maximum unoccupied eigenvalues for different kpoints (future references)
                    minimum_ocucupied_eigenvalues.append(min(occupied_eigenvalues))
                    maximum_unoccupied_eigenvalues.append(max(unoccupied_eigenvalues))


                else:
                    occupied_eigenvalues = bands["occupied"]
                    unoccupied_eigenvalues = bands["unoccupied"]
                    partial_occupied_eigenvalue = bands["partial"]
                    # Minimum occupied  and maximum unoccupied eigenvalues for different kpoints (future references)
                    minimum_ocucupied_eigenvalues.append(min(occupied_eigenvalues))
                    maximum_unoccupied_eigenvalues.append(max(unoccupied_eigenvalues))

                # Plot occupied eigenvalues for this k-point
                if len(occupied_eigenvalues) > 0:
                    ax.scatter([x_values[kpoint_idx]] * len(occupied_eigenvalues), occupied_eigenvalues,
                               color=plot_settings["occupied_color"], marker=plot_settings["occupied_marker"],**plot_settings['scatter_settings'])

                # Plot unoccupied eigenvalues for this k-point
                if len(unoccupied_eigenvalues) > 0:
                    ax.scatter([x_values[kpoint_idx]] * len(unoccupied_eigenvalues), unoccupied_eigenvalues,
                               color=plot_settings["unoccupied_color"], marker=plot_settings["unoccupied_marker"],**plot_settings['scatter_settings'])
                    
                # Plot partial occupied eigenvalues for this k-point
                if len(partial_occupied_eigenvalue) > 0:
                    ax.scatter([x_values[kpoint_idx]] * len(partial_occupied_eigenvalue), partial_occupied_eigenvalue,
                               color=plot_settings["partial_color"], marker=plot_settings["partial_marker"],**plot_settings['scatter_settings'])
                
            # Add a single legend for each spin plot
            if len(occupied_eigenvalues) > 0:
                ax.scatter([], [], color=plot_settings["occupied_color"], marker=plot_settings["occupied_marker"], label='Occupied',**plot_settings['scatter_settings'])
            if len(unoccupied_eigenvalues) > 0:
                ax.scatter([], [], color=plot_settings["unoccupied_color"], marker=plot_settings["unoccupied_marker"], label='Unoccupied',**plot_settings['scatter_settings'])
            if len(partial_occupied_eigenvalue) > 0:
                ax.scatter([], [], color=plot_settings["partial_color"], marker=plot_settings["partial_marker"], label='Partial occupied',**plot_settings['scatter_settings'])
                 
            # Set x-ticks and labels
            ax.set_xticks(x_values)  # The x-values are the indices of the k-points
            ax.set_xticklabels(_x_labels, rotation=0.0, ha="right")  # Apply the formatted labels

            ax.set_xlabel(plot_settings["xlabel"],size=plot_settings["label_size"])
            ax.set_ylabel(plot_settings["ylabel"],size=plot_settings["label_size"])

            # Set titles of the subplots
            if spin_key == 'spin 1':
                title = plot_settings["title_names"]["up"]
                if num_spins == 1:
                    title= None
            elif spin_key == 'spin 2':
                title = plot_settings["title_names"]["down"]
            else:
                raise ValueError(f"Error: Invalid spin key '{spin_key}' provided for title.")
            ax.set_title(
                        title,
                        fontdict= plot_settings["fontdict_title"]
                        )
            
            # If the energies are referenced to energy fermi
            y_value_VBM = VBM - VBM if fermi_energy_reference else VBM
            y_value_CBM = CBM - VBM if fermi_energy_reference else CBM
            
            # Set the y-axis limit if provided
            if y_limit == "(VBM-1.5,CBM+1.5)":
                y_limit = (y_value_VBM - 1.5,y_value_CBM +1.5)
                ax.set_ylim(y_limit)
            else:
                ax.set_ylim(y_limit)
            
            # Optionally plot valence and conduction band lines with custom line styles
            if plot_settings["show_vbm_cbm"]:
                ax.axhline(y=y_value_VBM, color=plot_settings["vbm_color"], linestyle=plot_settings["vbm_line_style"])
                ax.axhline(y=y_value_CBM, color=plot_settings["cbm_color"], linestyle=plot_settings["cbm_line_style"])

            # Legend location    
            ax.legend(loc=plot_settings["legend_loc"])

            # Conditionally display the shaded region VB and CB
            if show_fill_up:
                ax.axhspan(
                    ymin=min(minimum_ocucupied_eigenvalues),
                    ymax=y_value_VBM,
                    color = plot_settings["fill_up_color_vb"] ,
                    alpha = plot_settings["fill_up_alpha"]  # Should use plot_settings here
                    )
                ax.axhspan(
                    ymin=y_value_CBM,
                    ymax=max(maximum_unoccupied_eigenvalues),
                    color = plot_settings["fill_up_color_cb"],  # Should use plot_settings here
                    alpha = plot_settings["fill_up_alpha"]  # Should use plot_settings here
                    )   
        
        # Adjust spacing between subplots for better readability
        plt.tight_layout()  
        plt.close()
        return fig


    @staticmethod
    def get_plot_KS_eigenvalues(
                                vasprun_path: str,
                                VBM:float,
                                CBM:float,
                                y_limit:tuple|None="(VBM-1.5,CBM+1.5)",
                                fermi_energy_reference:bool = True,
                                show_fill_up:bool = True,
                                **plot_setting
                                )-> plt.Figure:
        """
        A static method to plot Kohn-Sham eigenvalues.

        Parameters:
        ----------
        vasprun_path : str
            Path to the VASP output file (vasprun.xml) to analyze.
        VBM : float
            Value of the Valence Band Maximum (VBM).
        CBM : float
            Value of the Conduction Band Minimum (CBM).
        y_limit : tuple, optional
            Tuple to specify the y-axis limits for the plot (default: None).
        fermi_energy_reference : bool, optional
            Whether to use the Fermi energy as the reference energy (default: True).
        show_fill_up : bool, optional
            Whether to visually highlight the occupied states (default: False).
        **plot_setting : dict
            Additional settings for customizing the plot (e.g., colors, line styles).

        Returns:
        -------
        None
            This method generates and displays the plot but does not return a value.
        """
        
        # Creat an instance VaspRunAnalysis class
        vasprun_analysis = VaspRunAnalysis(vasprun_path=vasprun_path)
        eigenvalues_dict, occupancy_dict = vasprun_analysis.get_Kohn_Sham_eigenvalues_and_occupancy()
        kpoints_dict = vasprun_analysis.get_kpoint_values()

        # Create an instance of PlotEigenvalue class 
        plotter = PlotKohnShamEigenvalue(eigenvalues_dict, occupancy_dict, kpoints_dict)
        fig = plotter.plot_KS_eigenvalues(VBM=VBM,CBM=CBM,y_limit=y_limit,fermi_energy_reference=fermi_energy_reference,show_fill_up=show_fill_up,**plot_setting)
        return fig






