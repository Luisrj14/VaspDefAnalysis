from ase.io import read
import numpy as np
from ase import Atoms

#from tools_packages.utils.tool_pool import find_relative_distance_from_poscar_with_respect_to_position
from VaspDefAnalysis.utils.tool_pool import find_indexs_positions_distances_symbols_inside_raduis

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

    def ion_neighbor_indices_to_defects(self, add_self_index=True) -> list:
        """
        Retrieves the indices of all neighboring atoms to the defect site, ensuring no duplicate indices.

        This function iterates through each defect position, calculates the neighboring atoms' indices,
        and appends them to a list. Finally, it removes duplicate indices and returns the unique set.

        Parameters:
        -----------
        add_self_index : bool, optional
            If True, includes the defect's own index in the list (default: True).

        Returns:
        --------
        list
            A list of unique indices of neighboring atoms to the defect.
        """

        all_neighbor_indices = []  # List to store all neighbor indices for each defect position

        # Get defect positions from the structure
        defect_positions = self.get_defect_positions()

        # Ensure radius is set, or provide a default fallback value
        if self.radius is None:
            raise ValueError("A value for the radius parameter is required to detect neighbors inside this radius.")

        # Loop through each defect position to find neighbors
        for def_pos in defect_positions:
            # Get indices of neighbors within the radius from the defect position
            neighbor_indices = find_indexs_positions_distances_symbols_inside_raduis(
                structure=self.defect_structure,   
                radius_centered_in_position=def_pos,  
                radius=self.radius
            )["indexs"]

            if add_self_index:
                # Identify the defect's own index
                for index_defect, pos in enumerate(self.defect_structure.get_positions()):
                    if np.linalg.norm(pos - def_pos) <= self.tolerance:
                        all_neighbor_indices.append(index_defect)

            # Append neighbor indices
            all_neighbor_indices.extend(neighbor_indices)

        # Remove duplicates using a set
        return list(set(all_neighbor_indices))
    
    def transform_to_fraction_coordinate(self, atom_position: np.ndarray):
        """
        Convert a Cartesian coordinate to fractional coordinates.

        Parameters:
        atom_position (np.ndarray): A 3D vector representing the atomic position in Cartesian coordinates.

        Returns:
        np.ndarray: The fractional coordinates of the given atomic position.
        """
        if len(atom_position) != 3: 
            raise ValueError("The atomic position must be a 3D vector or list.")
        
        # Convert Cartesian coordinates to fractional coordinates using the inverse of the cell matrix
        return np.dot(np.linalg.inv(self.perfect_structure.cell), atom_position) 

    
    def transform_cartesian_coordinate(self, atom_position: np.ndarray):
        """
        Convert a fractional coordinate to Cartesian coordinates.

        Parameters:
        atom_position (np.ndarray): A 3D vector representing the atomic position in fractional coordinates.

        Returns:
        np.ndarray: The Cartesian coordinates of the given atomic position.

        """
        if len(atom_position) != 3: 
            raise ValueError("The atomic position must be a 3D vector or list.")
        
        # Convert fractional coordinates to Cartesian coordinates using the cell matrix
        return np.dot(self.defect_structure.cell, atom_position)

    
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
            print(f'Vacancy: {vacancies[0][0]}, in the position: {self.transform_to_fraction_coordinate(vacancies[0][1])}')

        elif len(vacancies) == 0 and len(interstitials) == 1 and len(substitutions) == 0:
            print(f'Interstitial: {interstitials[0][0]}, in the position: {self.transform_to_fraction_coordinate(interstitials[0][1])}')

        elif len(vacancies) == 0 and len(interstitials) == 0 and len(substitutions) == 1:
            print(f'Substitution: {substitutions[0][0]} replaced by {substitutions[0][1]}, in the position: {self.transform_to_fraction_coordinate(substitutions[0][2])}')

        # Complex defects
        else:
            print('Complex:')
            if len(vacancies) != 0:
                print('Vacancy:' if len(vacancies) == 1 else 'Vacancies:')
                for vacancy in vacancies:
                    print(f'{vacancy[0]} at {self.transform_to_fraction_coordinate(vacancy[1])}')

            if len(interstitials) != 0:
                print('Interstitial:' if len(interstitials) == 1 else 'Interstitials:')
                for interstitial in interstitials:
                    print(f'{interstitial[0]} at {self.transform_to_fraction_coordinate(interstitial[1])}')

            if len(substitutions) != 0:
                print('Substitution:' if len(substitutions) == 1 else 'Substitutions:')
                for substitution in substitutions:
                    print(f'{substitution[0]} replaced by {substitution[1]} at {self.transform_to_fraction_coordinate(substitution[2])}')
