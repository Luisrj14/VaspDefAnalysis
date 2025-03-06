import numpy as np

from VaspDefAnalysis.read_vasp.vasprun_analysis import VaspRunAnalysis
from VaspDefAnalysis.plotter.tool_plotter import classify_eigenvalues,generate_fraction_labels_for_kpoints

class EigenvaluesAnalysis:
    def __init__(self, vasprun_path: str):
        self.vasprun_path = vasprun_path

        # Creat an instance VaspRunAnalysis class
        vasprun_analysis = VaspRunAnalysis(vasprun_path=self.vasprun_path)
        self.eigenvalues_dict, self.occupancy_dict = vasprun_analysis.get_Kohn_Sham_eigenvalues_and_occupancy()

        #self.classified_eigenvalues, self.classified_eigenvalues_band_index = classify_eigenvalues(eigenvalues_dict=self.eigenvalues_dict,occupancy_dict=self.occupancy_dict)

    def classify_eigenvalues(self):

            """
            Classify eigenvalues based on their occupancy into three categories: 
            occupied, unoccupied, and partially occupied.

            Parameters
            ----------
            eigenvalues_dict : dict
                A dictionary containing eigenvalues for different spins and k-points. 
                Structure: {spin_key: {kpoint_key: eigenvalue_list}}.

            occupancy_dict : dict
                A dictionary containing occupancies for different spins and k-points. 
                Structure: {spin_key: {kpoint_key: occupancy_list}}.

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
            - This function assumes that `eigenvalues_dict` contains the eigenvalues for each spin and k-point.
            - It also assumes that `occupancy_dict` provides the corresponding occupancy values.
            - If a k-point has no occupancy information in `occupancy_dict`, an empty list is used by default.

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
            
            classified_eigenvalues_band_index:
            {
                "spin 1": {
                    "k-point 1": {
                        "occupied": [...],      # List of index occupied eigenvalues
                        "unoccupied": [...],    # List of index unoccupied eigenvalues
                        "partial": [...]        # List of index partially occupied eigenvalues
                    },
                    ...
                },
                ...
            }
            """
            classified_eigenvalues = {}
            classified_eigenvalues_band_index = {}
            for spin_key, eigenvalues in self.eigenvalues_dict.items():
                classified_eigenvalues[spin_key] = {}
                classified_eigenvalues_band_index[spin_key]= {}
                for kpoint_key, eigenval_list in eigenvalues.items():
                    classified_eigenvalues[spin_key][kpoint_key] = {
                        "occupied": [],
                        "unoccupied": [],
                        "partial": []
                    }

                    classified_eigenvalues_band_index[spin_key][kpoint_key] = {
                        "occupied": [],
                        "unoccupied": [],
                        "partial": []
                    }
                    band_index = 1 
                    occupancy_list = self.occupancy_dict[spin_key].get(kpoint_key, [])
                    for eigenval, occupancy in zip(eigenval_list, occupancy_list):
                        if occupancy >= 0.9:
                            classified_eigenvalues[spin_key][kpoint_key]["occupied"].append(eigenval)
                            classified_eigenvalues_band_index[spin_key][kpoint_key]["occupied"].append(band_index)
                        elif occupancy <= 0.1:
                            classified_eigenvalues[spin_key][kpoint_key]["unoccupied"].append(eigenval)
                            classified_eigenvalues_band_index[spin_key][kpoint_key]["unoccupied"].append(band_index)
                        else:
                            classified_eigenvalues[spin_key][kpoint_key]["partial"].append(eigenval)
                            classified_eigenvalues_band_index[spin_key][kpoint_key]["partial"].append(band_index)
                        band_index +=1
            return classified_eigenvalues,classified_eigenvalues_band_index

    
    def electrinic_state_inf(self):
        for spin_idx, (spin_key, kpoint_keys) in enumerate(self.classified_eigenvalues.items()):
            for kpoint_idx, (kpoint_key, bands) in enumerate(kpoint_keys.items()):
                occupied_eigenvalues = bands["occupied"]
                unoccupied_eigenvalues = bands["unoccupied"]
                partial_occupied_eigenvalue = bands["partial"]

                # Add text labels for each eigenvalue
                for eig in self.eigenvalues_dict[spin_key][kpoint_key]:
                    pass
                    

            
