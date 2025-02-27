import numpy as np
from ase import Atoms

"""
Here there are some random tools
"""
def find_relative_distance_from_poscar_with_respect_to_position(structure: Atoms, position: np.ndarray) -> np.ndarray:
    """
    Finds distances of all atoms in the supercell with respect to a given position.

    Parameters:
    ----------
    structure : ase.Atoms
        The supercell structure as an ASE Atoms object, which contains atomic positions and metadata.

    position : np.ndarray
        The reference position in Cartesian coordinates, a 1D array (e.g., [x, y, z]).

    Returns:
    -------
    np.ndarray
        An array of distances (float values) from the reference position to all atoms in the supercell.
    """

    # Validate that the position is a 1D array of length 3
    if not isinstance(position, np.ndarray) or position.shape != (3,):
        raise ValueError("Position must be a 1D array with 3 elements representing [x, y, z].")

    # Extract atomic positions from the supercell object
    positions = structure.get_positions()  # Returns an (N, 3) array of positions for N atoms

    # Compute displacement vectors from the reference position to each atom
    displacements = positions - position  # Broadcast the position to (N, 3) for subtraction

    # Calculate the Euclidean norm (distance) for each displacement vector
    distances = np.linalg.norm(displacements, axis=1)  # Resulting distances in a 1D array

    return distances    

def find_indexs_positions_distances_symbols_inside_raduis(structure: Atoms,
                                                           radius_centered_in_position: np.ndarray,
                                                           radius:float
                                                           ):
           
    """
    Finds all atoms within a given radius from a specified position.

    Parameters:
    ----------
    structure : ase.Atoms
        The supercell structure as an ASE Atoms object containing atomic positions and metadata.

    radius_centered_in_position : array-like
        The center of the radius in Cartesian coordinates (1D array of length 3, e.g., [x, y, z]).

    radius : float
        The radius within which atoms are considered neighbors.

    Returns:
    -------
    tuple
        A tuple containing:
        - index_inside_radius (list[int]): Indices of atoms inside the radius.
        - distances_inside_radius (list[float]): Distances to atoms inside the radius.
        - neighbors_inside_radius (list[np.ndarray]): Positions of the atoms inside the radius.
    """
    
    # Calculate distances from the center to all atoms
    distances = find_relative_distance_from_poscar_with_respect_to_position(structure, radius_centered_in_position)
   
    index_inside_radius = []  # Indices of atoms within the radius
    distances_inside_radius = []  # Distances of atoms within the radius

    # Loop through the distances and check which are within the radius
    index = 0
    for dis in distances:
        if 0.0 < dis < radius:
            distances_inside_radius.append(dis)
            index_inside_radius.append(index)
        index+=1

    # Get the positions of the neighbors from the structure using the identified indices
    neighbors_inside_radius = [structure.get_positions()[i] for i in index_inside_radius]

    # Get the chemical symbols of the neighbors
    chemical_symbols = [structure.get_chemical_symbols()[i] for i in index_inside_radius]

    # Ensure at least one neighbor is found
    if not index_inside_radius:
        raise ValueError("No neighbors found within the given radius.")
            
    return {
            "indexs":index_inside_radius,
            "distances":distances_inside_radius,
            "posiitions":neighbors_inside_radius,
            "symbols":chemical_symbols}

class SIUnitConverter:
    """A class to provide SI multiples and submultiples for a given value and unit."""

    def __init__(self, value, unit):
        self.value = value
        self.unit = unit

        # SI prefixes with their corresponding factors and symbols
        self.prefixes = {
            "yotta": {"value": 1e24, "symbol": "Y"},   # Y
            "zetta": {"value": 1e21, "symbol": "Z"},   # Z
            "exa": {"value": 1e18, "symbol": "E"},     # E
            "peta": {"value": 1e15, "symbol": "P"},    # P
            "tera": {"value": 1e12, "symbol": "T"},    # T
            "giga": {"value": 1e9, "symbol": "G"},     # G
            "mega": {"value": 1e6, "symbol": "M"},     # M
            "kilo": {"value": 1e3, "symbol": "k"},     # k
            
            "milli": {"value": 1e-3, "symbol": "m"},   # m
            "micro": {"value": 1e-6, "symbol": "µ"},   # µ
            "nano": {"value": 1e-9, "symbol": "n"},    # n
            "pico": {"value": 1e-12, "symbol": "p"},   # p
            "femto": {"value": 1e-15, "symbol": "f"},  # f
            "atto": {"value": 1e-18, "symbol": "a"},   # a
            "zepto": {"value": 1e-21, "symbol": "z"},  # z
            "yocto": {"value": 1e-24, "symbol": "y"}   # y
        }

    def convert(self, prefix):
        """Convert the value to a specified SI prefix."""

        if prefix == None:
            return self.value,f'{self.unit}'
        else:
            if prefix in self.prefixes:
                converted_value = self.value / self.prefixes[prefix]["value"]
                symbol = self.prefixes[prefix]["symbol"]
                return converted_value,f"{symbol}{self.unit}"
            else:
                return "Invalid SI prefix." 

def find_index_position_symbol_of_most_center_atom(structure: Atoms, species: str = None) -> dict:
    """
    Find the index, position, and symbol of the atom closest to the centroid of the structure.
    Optionally, limit the search to a specific atomic species.

    Parameters:
    ----------
    structure : Atoms
        An ASE Atoms object representing the atomic structure.
    species : str, optional
        The chemical symbol of the atomic species to consider. If None, considers all atoms.

    Returns:
    -------
    dict
        A dictionary containing the index, position, and symbol of the closest atom.
    """
    # Get atomic positions and symbols
    positions = structure.get_scaled_positions()
    symbols = np.array(structure.get_chemical_symbols())
    
    # Compute centroid (geometric center)
    centroid = np.mean(positions, axis=0)
    
    # Filter indices for the selected species
    if species is not None:
        indices = np.where(symbols == species)[0]
        if len(indices) == 0:
            raise ValueError(f"No atoms of species '{species}' found in the structure.")
    else:
        indices = np.arange(len(structure))
    
    # Compute distances from centroid for the selected indices
    distances = np.linalg.norm(positions[indices] - centroid, axis=1)
    
    # Find the index of the closest atom among the selected species
    min_index = indices[np.argmin(distances)]
    
    # Return central atom information
    return {
        "index": min_index,
        "position": structure.get_positions()[min_index],
        "symbol": symbols[min_index]
    }

#def find_indexs_positions_distances_symbols_to_neighbors(structure: Atoms, 
#                                                        neighbors_to_position: np.ndarray, 
#                                                        tolerance:float=1e-1,
#                                                        add_neighbors_up:int = 1
#                                                        ):
#    """
#    Finds the neighbors of a defect site based on the minimum non-zero distance.
#
#    Parameters:
#    ----------
#    structure : ase.Atoms
#        The supercell structure as an ASE Atoms object containing atomic positions and metadata.
#
#    neighbors_to_position : array-like
#        The position of the defect in Cartesian coordinates (1D array of length 3, e.g., [x, y, z]).
#
#    tolerance : float, optional
#        The tolerance value to account for numerical errors when comparing distances (default is 1e-5).
#
#    add_neighbors_up : int, optional
#        Number of neighbor distances to consider (default is 1, which finds the first nearest neighbors).
#
#    Returns:
#    -------
#    tuple
#        A tuple containing:
#        - index_neighbors (list[int]): Indices of the neighbors.
#        - distances_neighbors (list[float]): Distances to the neighbors.
#        - neighbors_positions (list[np.ndarray]): Positions of the neighbors.
#    """
#    distances = find_relative_distance_from_poscar_with_respect_to_position(structure, neighbors_to_position)
#
#    # Ensure there are at least two unique distances (i.e., the defect site and at least one neighbor)
#    if len(distances) < 2:
#        raise ValueError("No neighbors found.")  # Raise an error if no neighbors are found
#    
#    # Remove all occurrences of 0.0 (the reference position itself)
#    #distances = distances[distances != 0.0]
#    
#    # Sort the distances to find the nearest ones
#    distances_sorted = sorted(distances)
#
#    # Identify unique neighbor distances within the tolerance range
#    neighbors_distances = [distances_sorted[0]]
#    distances_try = distances_sorted[0]
#    for i in range(len(distances_sorted)-1):
#        # Check if the next distance is sufficiently different from the current one
#        if abs(distances_sorted[i+1] - distances_try) >= tolerance:
#            neighbors_distances.append(distances_sorted[i+1])
#            distances_try = distances_sorted[i+1]
#    
#    index_neighbors = []  # Store indices of neighbors
#    distance_neighbors = []  # Store distances of neighbors
#
#    # Loop through the specified number of neighbors
#    for add_up in range(add_neighbors_up):
#        index = 0
#        for dis in distances:
#            if abs(dis - neighbors_distances[add_up]) <= tolerance:
#                index_neighbors.append(index)
#                distance_neighbors.append(dis)
#
#            index+=1
#        
#        # Get the positions of the neighbors from the structure using the identified indices
#        neighbors_posiitions = [structure.get_positions()[i] for i in index_neighbors]
#
#        # Get the chemical symbols of the neighbors
#        chemical_symbols = [structure.get_chemical_symbols()[i] for i in index_neighbors]
#    
#    neighbors_imformation = {"indexs":index_neighbors,"distances":distance_neighbors,"positions":neighbors_posiitions,"symbols":chemical_symbols}
#    return neighbors_imformation