import numpy as np

from VaspDefAnalysis.read_vasp.vasprun_analysis import VaspRunAnalysis
from VaspDefAnalysis.plotter.tool_plotter import classify_eigenvalues,generate_fraction_labels_for_kpoints

class EigenvaluesAnalysis:
    def __init__(self, vasprun_path: str):
        self.vasprun_path = vasprun_path

        # Creat an instance VaspRunAnalysis class
        vasprun_analysis = VaspRunAnalysis(vasprun_path=self.vasprun_path)
        self.eigenvalues_dict, self.occupancy_dict = vasprun_analysis.get_Kohn_Sham_eigenvalues_and_occupancy()

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
    
    def electrinic_state_inf(self):
        classified_eigenvalues = classify_eigenvalues() 
        for spin_idx, (spin_key, kpoint_keys) in enumerate(classified_eigenvalues.items()):
            # Save the minimum occupied  and maximum unoccupied eigenvalues for different kpoints (future references)
            minimum_ocucupied_eigenvalues = []
            maximum_unoccupied_eigenvalues = []
            for kpoint_idx, (kpoint_key, bands) in enumerate(kpoint_keys.items()):
                occupied_eigenvalues = bands["occupied"]
                unoccupied_eigenvalues = bands["unoccupied"]
                partial_occupied_eigenvalue = bands["partial"]
                # Minimum occupied  and maximum unoccupied eigenvalues for different kpoints (future references)
                minimum_ocucupied_eigenvalues.append(min(occupied_eigenvalues))
                maximum_unoccupied_eigenvalues.append(max(unoccupied_eigenvalues))
                band_index = 1
                # Add text labels for each eigenvalue
                for eig in self.eigenvalues_dict[spin_key][kpoint_key]:
                    
                    band_index += 1 
            
