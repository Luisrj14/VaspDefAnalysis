from ase.io import read
import numpy as np
from ase import Atoms

#from tools_packages.utils.tool_pool import find_relative_distance_from_poscar_with_respect_to_position
from VaspDefAnalysis.utils.tool_pool import find_indexs_positions_distances__symbols_inside_raduis

class StructureComparator:
    """
    A class to perform defect analysis by comparing perfect and defect structures.
    Identifies vacancies, interstitials, and substitutions, and complex.
    
    """
    def __init__(self,
                 perfect_structure_file:str,
                 defect_structure_file:str,
                 radius:float=None,
                 tolerance=2e-1,
                 ):

        """
        Initializes the DefectAnalisys class with the perfect and defect structures.

        Parameters:
        ----------
        perfect_structure_file : str
            Path or file to the perfect structure, in VASP format.
        defect_structure_file : str
            Path or file to the defect structure, in VASP format without relaxing.
        tolerance : float, optional
            The tolerance for identifying defects (default is 1e-3).
        add_neighbors_up : int, optional
            The number of additional neighbors to consider (default is 1).
        """
        self.perfect_structure = read(perfect_structure_file, format='vasp') if isinstance(perfect_structure_file, str) else perfect_structure_file
        self.defect_structure = read(defect_structure_file, format='vasp')  if isinstance(defect_structure_file, str) else defect_structure_file
        self.tolerance = tolerance
        self.radius = radius

    def structure_diff(self):
        """
        Compares the perfect and defect structures to identify defects (including complex defects).

        Returns:
        -------
        dict
            A dictionary summarizing detected defects:
            {
                "vacancies": [(symbol, position), ...],
                "interstitials": [(symbol, position), ...],
                "substitutions": [(original_symbol, new_symbol, position), ...],
            }
        """
        # Read the structures
        perfect = self.perfect_structure 
        defect = self.defect_structure 

        # Tolerance
        tolerance = self.tolerance

        # Get positions and symbols
        perfect_positions = perfect.get_positions()
        defect_positions = defect.get_positions()
        perfect_symbols = perfect.get_chemical_symbols()
        defect_symbols = defect.get_chemical_symbols()

        # Initialize defect lists
        vacancies = []
        interstitials = []
        substitutions = []

        # Analyze vacancies: Atoms in perfect structure not found in defect structure
        for perfect_symbol, perfect_position in zip(perfect_symbols, perfect_positions):
            distances = np.linalg.norm(defect_positions - perfect_position, axis=1)
            if np.all(distances >= tolerance):  
                vacancies.append((perfect_symbol, perfect_position))

        # Analyze interstitials: Atoms in defect structure not found in perfect structure
        for defect_symbol, defect_position in zip(defect_symbols, defect_positions):
            distances = np.linalg.norm(perfect_positions - defect_position, axis=1)
            if np.all(distances >= tolerance):  # No matching position in perfect
                interstitials.append((defect_symbol, defect_position))

        # Analyze substitutions: Atoms in perfect structure replaced with different atoms in defect structure
        for perfect_symbol, perfect_position in zip(perfect_symbols, perfect_positions):
            distances = np.linalg.norm(defect_positions - perfect_position, axis=1)
            closest_index = np.argmin(distances)
            if distances[closest_index] <= tolerance:  # Close match found
                defect_symbol = defect_symbols[closest_index]
                if perfect_symbol != defect_symbol:
                    substitutions.append((perfect_symbol, defect_symbol, defect_positions[closest_index]))

        # Return all detected defects
        return {
            "vacancies": vacancies,
            "interstitials": interstitials,
            "substitutions": substitutions,
        }


    def get_defect_positions(self)->list:
        """
        Extracts the positions of all defects (vacancies, interstitials, and substitutions) 
        from the analyze_defect function.

        Returns:
        -------
        list
            A list of defect positions: [np.array(position), ...]
        """
        defects = self.structure_diff()

        defect_positions = []

        # Extract the positions of vacancies, interstitials, and substitutions
        for defect_type in defects.values():
            for defect in defect_type:
                # In vacancies and interstitials, defect[1] is the position
                # In substitutions, defect[2] is the position
                if len(defect) == 2:  # Vacancy or Interstitial
                    defect_positions.append(defect[1])
                elif len(defect) == 3:  # Substitution
                    defect_positions.append(defect[2])
        return defect_positions

    def ion_neighbor_indices_to_defects(self)-> list:
        """
        Retrieves the indices of all neighboring atoms to the defect site, ensuring no duplicate indices.

        This function iterates through each defect position, calculates the neighboring atoms' indices,
        and appends them to a list. Finally, it removes duplicate indices and returns the unique set.

        Returns:
        -------
        list
            A list of unique indices of neighboring atoms to the defect.
        """

        all_neighbor_indices = []  # List to store all the neighbor indices for each defect position

        # Get the defect positions from the structure
        defect_positions = self.get_defect_positions()

        if self.radius == None:
            raise ValueError(f"It requires a value for the radius parameter to detect the neighbor. By default, the value is set to {self.radius}.")

        
        # Loop through each defect position
        for def_pos in defect_positions:
            # Get indices, positions, and distances of the neighbors to the current defect position
            neighbor_indices = find_indexs_positions_distances__symbols_inside_raduis(
                structure=self.defect_structure,        # The defect structure
                radius_centered_in_position=def_pos,          # The current defect position
                radius=self.radius
                )["indexs"]

            # Add the susitutional or interestitial indexes of the defect.
            index_defect = 0 
            for pos in self.defect_structure.get_positions():
                if np.linalg.norm(pos - def_pos) <= self.tolerance:
                    # Append the indices of neighbors to the list
                    all_neighbor_indices.append([index_defect])
                    
                index_defect+=1

            # Append the indices of neighbors to the list
            all_neighbor_indices.append(neighbor_indices)
        
        # Flatten the list of lists containing neighbor indices into a single list
        all_neighbor_indices_concatenate = np.concatenate(all_neighbor_indices).tolist()

        # Create a set to track the elements
        set_object = set()  

        # Remove duplicates by checking if the item has been seen before
        # If it hasn't been seen, add it to the result list and mark it as seen
        neighbor_indices_without_repetation = [item for item in all_neighbor_indices_concatenate if item not in set_object and not set_object.add(item)]
        
        # Return the list of unique neighbor indices
        return neighbor_indices_without_repetation
    
    @staticmethod
    def get_ion_neighbor_indices_to_defects(perfect_structure_file:str,
                                            defect_structure_file:str,
                                            radius:float,
                                            tolerance:float=1e-3,
                                            ):
        """
        Static method to retrieve neighbor indices for defect analysis.

        Parameters:
        ----------
        perfect_structure_file : str
            Path to the perfect structure.
        defect_structure_file : str
            Path to the defect structure without relaxing.
        tolerance : float
            Tolerance for identifying neighbors.
        add_neighbors_up : int
            Additional neighbors to consider.

        Returns:
        -------
        list
            Unique neighbor indices.
        """
        # Analyze defect
        defect_analyzed = StructureComparator(perfect_structure_file=perfect_structure_file,
                                         defect_structure_file=defect_structure_file,
                                         radius=radius,
                                         tolerance=tolerance)
        return defect_analyzed.ion_neighbor_indices_to_defects()
    
    def get_defect_information(self):

        """
        Prints detailed information about the detected defects.
        """
        
        analyze_defects = self.structure_diff()
        vacancies = analyze_defects['vacancies']
        interstitials = analyze_defects['interstitials']
        substitutions = analyze_defects['substitutions']

        # Check for single defect cases
        if len(vacancies) == 1 and len(interstitials) == 0 and len(substitutions) == 0:
            print(f'Vacancy: {vacancies[0][0]}, in the position: {vacancies[0][1]}')

        elif len(vacancies) == 0 and len(interstitials) == 1 and len(substitutions) == 0:
            print(f'Interstitial: {interstitials[0][0]}, in the position: {interstitials[0][1]}')

        elif len(vacancies) == 0 and len(interstitials) == 0 and len(substitutions) == 1:
            print(f'Substitution: {substitutions[0][0]} replaced by {substitutions[0][1]}, in the position: {substitutions[0][2]}')

        # Complex defects
        else:
            print('Complex:')
            if len(vacancies) != 0:
                print('Vacancy:' if len(vacancies) == 1 else 'Vacancies:')
                for vacancy in vacancies:
                    print(f'{vacancy[0]} at {vacancy[1]}')

            if len(interstitials) != 0:
                print('Interstitial:' if len(interstitials) == 1 else 'Interstitials:')
                for interstitial in interstitials:
                    print(f'{interstitial[0]} at {interstitial[1]}')

            if len(substitutions) != 0:
                print('Substitution:' if len(substitutions) == 1 else 'Substitutions:')
                for substitution in substitutions:
                    print(f'{substitution[0]} replaced by {substitution[1]} at {substitution[2]}')
