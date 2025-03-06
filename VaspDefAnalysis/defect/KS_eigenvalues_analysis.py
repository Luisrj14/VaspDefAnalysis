import numpy as np

from VaspDefAnalysis.read_vasp.vasprun_analysis import VaspRunAnalysis
from VaspDefAnalysis.plotter.tool_plotter import classify_eigenvalues
class EigenvaluesAnalysis:
    def __init__(self, vasprun_path: str):
        self.vasprun_path = vasprun_path

        # Creat an instance VaspRunAnalysis class
        vasprun_analysis = VaspRunAnalysis(vasprun_path=self.vasprun_path)
        self.eigenvalues_dict, self.occupancy_dict = vasprun_analysis.get_Kohn_Sham_eigenvalues_and_occupancy()
        
    def classify_eigenvalues(self)-> tuple:
        """
        Output
        --------------
        classified_eigenvalues:
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
        return classify_eigenvalues(eigenvalues_dict=self.eigenvalues_dict,occupancy_dict=self.occupancy_dict)

    
    def electrinic_state_inf(self):
        for spin_idx, (spin_key, kpoint_keys) in enumerate(self.classified_eigenvalues.items()):
            for kpoint_idx, (kpoint_key, bands) in enumerate(kpoint_keys.items()):
                occupied_eigenvalues = bands["occupied"]
                unoccupied_eigenvalues = bands["unoccupied"]
                partial_occupied_eigenvalue = bands["partial"]

                # Add text labels for each eigenvalue
                for eig in self.eigenvalues_dict[spin_key][kpoint_key]:
                    pass
                    

            
