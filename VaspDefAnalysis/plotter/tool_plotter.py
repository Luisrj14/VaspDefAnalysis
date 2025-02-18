# Packages
from fractions import Fraction

"""
Here there are some tools for plotter class
"""

def generate_fraction_labels_for_kpoints(kpt_coords:list, 
                                 line_break="\n"):
        
        """
        Generate x-axis labels for k-points based on their fractional coordinates.

        Parameters
        ----------
        kpt_coords : list of list of float
            A list of k-point coordinates, where each k-point is represented as a list of three fractional values (e.g., [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]).
        line_break : str, optional
            The character to use for separating coordinates in the label. Default is a newline character ("\n").

        Returns
        -------
        list of str
            A list of formatted k-point labels. Each label is either:
            - "Γ" for the origin (0, 0, 0).
            - A string representation of the fractional coordinates, with each coordinate as a fraction (e.g., "1/2\n0\n0").
        """
        result = []
        for k in kpt_coords:
            x_label = []
            for i in k:
                frac = Fraction(i).limit_denominator(10)  # Use fraction with denominator up to 10
                if frac.numerator == 0:
                    x_label.append("0")
                else:
                    x_label.append(f"{frac.numerator}/{frac.denominator}")
            if x_label == ["0", "0", "0"]:
                result.append("Γ")
            else:
                result.append(line_break.join(x_label))
        return result


def classify_eigenvalues(eigenvalues_dict:dict, 
                         occupancy_dict:dict)-> dict:
        
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
        """
        classified_eigenvalues = {}
        for spin_key, eigenvalues in eigenvalues_dict.items():
            classified_eigenvalues[spin_key] = {}
            for kpoint_key, eigenval_list in eigenvalues.items():
                classified_eigenvalues[spin_key][kpoint_key] = {
                    "occupied": [],
                    "unoccupied": [],
                    "partial": []
                }
                occupancy_list = occupancy_dict[spin_key].get(kpoint_key, [])
                for eigenval, occupancy in zip(eigenval_list, occupancy_list):
                    if occupancy >= 0.9:
                        classified_eigenvalues[spin_key][kpoint_key]["occupied"].append(eigenval)
                    elif occupancy <= 0.1:
                        classified_eigenvalues[spin_key][kpoint_key]["unoccupied"].append(eigenval)
                    else:
                        classified_eigenvalues[spin_key][kpoint_key]["partial"].append(eigenval)
        return classified_eigenvalues