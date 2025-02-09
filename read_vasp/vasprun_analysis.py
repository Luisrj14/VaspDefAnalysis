import xml.etree.ElementTree as ET

class VaspRunAnalysis:
    """
    The VaspRunAnalysis class provides functionality to analyze VASP (Vienna Ab initio Simulation Package)
    run data stored in a `vasprun.xml` file. This file contains comprehensive output from VASP calculations 
    such as eigenvalues, k-points, and orbital weights. The class uses Python's xml.etree.ElementTree to 
    parse and extract necessary data for detailed analysis of simulation results.
    
    Initialize the VaspAnalysis class with the path to vasprun.xml file.
    
    Parameters:
    vasprun_path (str): Path to the vasprun.xml file
    """
    def __init__(self, vasprun_path: str):
        self.vasprun_path = vasprun_path

        # Transform tree and root in attributes to easier called
        self.tree = ET.parse(self.vasprun_path)
        self.root = self.tree.getroot()

    def get_ion_orbitals_total_weights(self) -> dict:
        """
        Extracts the orbital weights from vasprun.xml for specified ions.

        Parameters:
        representative_ions (int): Number of representative ions to include.

        Returns:
        dict: Dictionary with orbital weights for each spin and k-point.

        dict: A dictionary organized as {spin: {kpoint index: [[band 1][band 2]...[band N]]}}.
              Where [band 1] has the all ion orbital weights.   
        """
        weights = {}

        for spin in self.root.findall(".//set[@comment='spin1']") + self.root.findall(".//set[@comment='spin2']"):
            spin_key = spin.attrib['comment']
            weights[spin_key] = {}

            for kpoint_index, kpoint_set in enumerate(spin.findall("set")):
                kpoint_key = f"kpoint {kpoint_index + 1}"
                weights[spin_key][kpoint_key] = []

                for band in kpoint_set.findall(".//set"):
                    ions_orbital_total_weights = []

                    for r in band.findall(".//r"):
                        ion_orbital_weights = map(float, r.text.split())
                        total_weight = sum(ion_orbital_weights)
                        ions_orbital_total_weights.append(total_weight)

                    # Save in a list the total weights     
                    weights[spin_key][kpoint_key].append(ions_orbital_total_weights)
        
        return weights

    def get_Kohn_Sham_eigenvalues_and_occupancy(self) -> tuple[dict, dict]:
        """
        Extracts Kohn-Sham eigenvalues and occupancies from vasprun.xml.

        Returns:
        tuple[dict, dict]: Dictionaries for eigenvalues and occupancy by spin and k-point.

        dict: A dictionary organized as {spin: {kpoint index: eigenvalues}}.
        dict: A dictionary organized as {spin: {kpoint index: occupancies}}.
        """
        eigenvalues_dict = {}
        occupancy_dict = {}
        
        eigenvalues_data = self.root.find(".//eigenvalues/array/set")
        
        for spin_index, spin_set in enumerate(eigenvalues_data.findall("set")):
            spin_key = f"spin {spin_index + 1}"
            eigenvalues_dict[spin_key] = {}
            occupancy_dict[spin_key] = {}

            for kpoint_index, kpoint_set in enumerate(spin_set.findall("set")):
                kpoint_key = f"kpoint {kpoint_index + 1}"
                eigenvalues_dict[spin_key][kpoint_key] = []
                occupancy_dict[spin_key][kpoint_key] = []

                for band_data in kpoint_set.findall("r"):
                    eigenvalue, occupancy = map(float, band_data.text.split())
                    eigenvalues_dict[spin_key][kpoint_key].append(eigenvalue)
                    occupancy_dict[spin_key][kpoint_key].append(occupancy)
        
        return eigenvalues_dict, occupancy_dict

    def get_kpoint_values(self) -> dict:
        """
        Extracts the k-point values (e.g., K = (0, 0, 0)) from vasprun.xml and organizes them by spin and k-point index.

        Returns:
        dict: A dictionary organized as {spin: {kpoint index: kpoint value}}.
        """
        kpoints_data = self.root.find(".//varray[@name='kpointlist']")
        spins = self.root.findall(".//eigenvalues/array/set")

        organized_kpoints = {}

        for spin_index, spin_set in enumerate(spins):
            spin_key = f"spin {spin_index + 1}"
            organized_kpoints[spin_key] = {}

            for kpoint_index, kpoint_element in enumerate(kpoints_data.findall("v")):
                kpoint_values = tuple(map(float, kpoint_element.text.split()))
                kpoint_key = f"kpoint {kpoint_index + 1}"
                organized_kpoints[spin_key][kpoint_key] = kpoint_values

        return organized_kpoints