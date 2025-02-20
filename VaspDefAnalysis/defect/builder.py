from ase.build.supercells import make_supercell
from ase import io, Atoms
from numpy import ndarray
from ase.io import read

class Supercell:
    def __init__(self):
        """
        Initialize the Supercell class. Currently, no attributes are defined.
        """
        pass
    
    @staticmethod
    def io_read(structure:str):
        """
        Read the structure data from a given source.

        Parameters:
        -----------
        structure : str
            The file or source that contains the structure data to be read.

        Returns:
        --------
        Object
            The structure data read from the source, returned by the `io.read()` function.
        """
        return io.read(structure)

    @staticmethod
    def get_supercell(structure: str, transformation_matrix: ndarray, **settings) -> Atoms:
        """
        Generate a supercell from a given structure and transformation matrix.

        Parameters:
        -----------
        structure : str
            The structure to be used to generate the supercell, typically a path to a file.
        
        transformation_matrix : ndarray
            A matrix that defines how the structure should be transformed to create the supercell.

        **settings : keyword arguments
            Additional optional settings that may be passed to the `make_supercell` function.

        Returns:
        --------
        Atoms
            The generated supercell structure.
        """
        io_structure = Supercell.io_read(structure)  # Read the original structure
        return make_supercell(prim=io_structure, P=transformation_matrix, **settings) 
    
    @staticmethod
    def get_squart_shape_supercell(structure):
        pass

class MakeSimpleDefect:
    
    "Generate simples defect"

    def __init__(self, structure: str):
        """Initialize with a structure file (e.g., POSCAR, CIF)."""
        self.structure = structure
        self.atoms = io.read(structure) if isinstance(structure, str) else structure # Read the structure using ASE

    def make_vacancy(self, index: int) -> Atoms:
        """
        Create a vacancy by removing an atom at a given index.
        
        Parameters:
        ----------
        index : int
            Index of the atom to be removed.
        
        Return:
        ------
        Atoms object with a vacancy.
        """
        defect_structure = self.atoms.copy()
        print(defect_structure)
        del defect_structure[index]
        return defect_structure

    def make_interstitial(self, position: list, element: str) -> Atoms:
        """
        Create an interstitial defect by adding an extra atom at a given position.
        
        Parameters:
        ----------
        position : list
            List [x, y, z] where the atom should be inserted.
        element : str
            Chemical symbol of the interstitial atom.
        
        Return:
        ------
        Atoms object with an interstitial.
        """
        defect_structure = self.atoms.copy()
        defect_structure.append(Atoms(element, positions=[position]))
        return defect_structure

    def make_substitutional(self, index: int, new_element: str) -> Atoms:
        """
        Replace an atom with another element to create a substitutional defect.
        
        Parameters:
        ----------
        index : int
            Index of the atom to be replaced.
        new_element : str
            Chemical symbol of the replacing atom.
        
        Return:
        ------
        Atoms object with a substitutional defect.
        """
        defect_structure = self.atoms.copy()
        defect_structure[index].symbol = new_element
        return defect_structure


class MakeComplexDefect(MakeSimpleDefect):
    
    "Generate complex defect"

    def __init__(self, structure):
        super().__init__(structure)
    def make_divacancy(self, indices: list) -> Atoms:
        """
        Create a divacancy by removing two atoms at specified indices.
        
        Parameters:
        ----------
        indices : list of int
            List of two indices of atoms to be removed.
        
        Return:
        ------
        Atoms object with a divacancy.
        """
        if len(indices) != 2:
            raise ValueError("Divacancy requires exactly two atom indices.")
        
        defect_structure = self.atoms.copy()
        for index in sorted(indices, reverse=True):  # Reverse to avoid index shifting
            del defect_structure[index]
        
        return defect_structure

    def make_antisite(self, index1: int, index2: int) -> Atoms:
        """
        Create an antisite defect by swapping two atom types.
        
        Parameters:
        ----------
        index1 : int
            Index of the first atom.
        index2 : int
            Index of the second atom.
        
        Return:
        ------
        Atoms object with an antisite defect.
        """
        defect_structure = self.atoms.copy()
        element1 = defect_structure[index1].symbol
        element2 = defect_structure[index2].symbol

        defect_structure[index1].symbol = element2
        defect_structure[index2].symbol = element1

        return defect_structure

    def make_double_substitutional(self, indices: list, new_element: str) -> Atoms:
        """
        Create a double substitutional defect (like a dimer).
        
        Parameters:
        ----------
        indices : list of int
            List of two atom indices to be replaced.
        new_element : str
            Chemical symbol of the replacing atoms.
        
        Return:
        ------
        Atoms object with a double substitutional defect.
        """
        if len(indices) != 2:
            raise ValueError("Double substitution requires exactly two atom indices.")

        defect_structure = self.atoms.copy()
        for index in indices:
            defect_structure[index].symbol = new_element

        return defect_structure
    
    def make_antisite_vacancy(self, antisite_index: int, vacancy_index: int) -> Atoms:
        """
        Create an antisite-vacancy defect where one atom is swapped with another,
        and the second atom is removed, leaving a vacancy.
    
        Parameters:
        -----------
        antisite_index : int
            The index of the atom to be swapped with another atom at `vacancy_index`.
        
        vacancy_index : int
            The index of the atom that will be removed, creating a vacancy.
    
        Returns:
        --------
        Atoms
            A new atomic structure with the antisite-vacancy defect, where the atom at
            `antisite_index` has been swapped and the atom at `vacancy_index` has been removed.
        """
        defect_structure = self.atoms.copy()  # Create a copy of the atoms in the original structure
        element1 = defect_structure[antisite_index].symbol  # Get the symbol of the atom at antisite_index
        element2 = defect_structure[vacancy_index].symbol  # Get the symbol of the atom at vacancy_index
        
        # Swap atoms
        defect_structure[antisite_index].symbol = element2  # Replace element at antisite_index with element2
        
        # Remove the second atom
        del defect_structure[vacancy_index]  # Delete the atom at vacancy_index
        
        return defect_structure  # Return the modified defect structure

class Makedefect(MakeComplexDefect):
    def __init__(self, structure):
        super().__init__(structure)