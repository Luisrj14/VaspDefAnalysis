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
        return
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

    
    def get_electorinic_state_band_inf(self)-> str:
        classified_eigenvalues, classified_eigenvalues_band_index = self.classify_eigenvalues() 

        if len(classified_eigenvalues) != 2 :
            raise ValueError("This class EigenvaluesAnalysis just works with polarized spin using VASP")

        nband_up = len(classified_eigenvalues["spin 1"]["kpoint 1"]["occupied"]) + len(classified_eigenvalues["spin 1"]["kpoint 1"]["unoccupied"]) \
                + len(classified_eigenvalues["spin 1"]["kpoint 1"]["partial"]) 
        nband_up_occupied = len(classified_eigenvalues["spin 1"]["kpoint 1"]["occupied"])
        nband_up_unoccupied = len(classified_eigenvalues["spin 1"]["kpoint 1"]["unoccupied"])
        nband_up_partial = len(classified_eigenvalues["spin 1"]["kpoint 1"]["partial"])
        if nband_up_partial != 0:
           print(
                "There are partially occupied bands with (spin up)."
                "This may indicate a nearly closed state, which could be degenerate, allowing electrons to be in both states."
                )
        nband_down = len(classified_eigenvalues["spin 2"]["kpoint 1"]["occupied"]) + len(classified_eigenvalues["spin 2"]["kpoint 1"]["unoccupied"]) \
                + len(classified_eigenvalues["spin 2"]["kpoint 1"]["partial"]) 
        nband_down_occupied = len(classified_eigenvalues["spin 2"]["kpoint 1"]["occupied"])
        nband_down_unoccupied = len(classified_eigenvalues["spin 2"]["kpoint 1"]["unoccupied"])
        nband_down_partial = len(classified_eigenvalues["spin 2"]["kpoint 1"]["partial"])
        if nband_down_partial != 0:
           print("There are partially occupied bands with (spin down)."
               "This may indicate a nearly closed state, which could be degenerate, allowing electrons to be in both states."
                )
        nelectron = nband_up_occupied + nband_down_occupied + int((nband_down_partial+ nband_up_partial)/2)
        nkpoints = len(classified_eigenvalues["spin 1"])

        electronic_state_band_info= f"""
=================================================================
Number of bands (For each spin):                                           
   NBAND = {nband_up:<10}                                  
                                                           
Number of k-points                                         
   NKPOINT = {nkpoints:<10}                                
                                                           
Number of occupied bands (spin up)                         
   NBAND-OCC-UP = {nband_up_occupied:<10}                  
                                                           
Number of unoccupied bands (spin up)                       
   NBAND-UNOCC-UP = {nband_up_unoccupied:<10}

Number of partial occupied bands (spin up)                       
    NBAND-PAR-OCC-UP = {nband_up_partial:<10}    
                                                           
Number of occupied bands (spin down)                       
   NBAND-OCC-DOWN = {nband_down_occupied:<10}              
                                                           
Number of unoccupied bands (spin down)                     
   NBAND-UNOCC-DOWN = {nband_down_unoccupied:<10}   

Number of partial occupied bands (spin down)                       
   NBAND-PAR-OCC-DOWN = {nband_down_partial:<10}                                                          
                                                                       
Number of electrons                                                          
   NELECTRON = {nelectron:<10}                                                             
=================================================================
        """
        return electronic_state_band_info
            
