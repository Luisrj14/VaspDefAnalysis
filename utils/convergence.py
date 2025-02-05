import numpy as np

class ConvergenceTools:
    """A collection of tools for convergence plot"""

    @staticmethod
    def create_abs_diff_values(
        vector_1: np.ndarray,
        vector_2: np.ndarray = None
    ) -> np.ndarray:
        """
        Calculate relative values between elements of `vector_1` or between `vector_1` and `vector_2`.

        Parameters:
        vector_1 (np.ndarray): A one-dimensional NumPy array.
        vector_2 (np.ndarray, optional): A second one-dimensional array of the same length as `vector_1`.
                                         If not provided, the difference between consecutive elements of
                                         `vector_1` is returned.

        Returns:
        np.ndarray: An array of relative values.
        """
        # Validate input dimensions
        if vector_1.ndim != 1:
            raise ValueError("Expected a one-dimensional array for `vector_1`.")
        if vector_2 is not None and vector_2.ndim != 1:
            raise ValueError("Expected a one-dimensional array for `vector_2`.")

        if vector_2 is None:
            # Differences between consecutive elements in vector_1
            relative_vector = np.abs(np.diff(vector_1))
        else:
            if len(vector_1) != len(vector_2):
                raise ValueError("Vectors must be of the same length.")
            relative_vector = np.abs(vector_2 - vector_1)

        return relative_vector

    @staticmethod
    def find_convergence_values(
        relative_vector: np.ndarray,
        cut_off_values: np.ndarray,
        conv_criterion: float
    ) -> np.ndarray:
        """
        Find values from `cut_off_values` where the corresponding elements in `relative_vector` meet the convergence criterion.

        If `relative_vector` and `cut_off_values` have lengths that differ by 1, the last element of `cut_off_values`
        will be discarded to match the length of `relative_vector`.

        Parameters:
        relative_vector (np.ndarray): A one-dimensional NumPy array of relative values.
        cut_off_values (np.ndarray): A one-dimensional NumPy array of values to be filtered based on the convergence criterion.
        conv_criterion (float): The threshold for convergence.

        Returns:
        np.ndarray: An array containing the values from `cut_off_values` where the corresponding value in `relative_vector` is <= `conv_criterion`.
        """
        if relative_vector.ndim != 1:
            raise ValueError("Expected a one-dimensional array for `relative_vector`.")
        if cut_off_values.ndim != 1:
            raise ValueError("Expected a one-dimensional array for `cut_off_values`.")

        if len(relative_vector) == len(cut_off_values):
            # Use vectorized operation to filter values
            save_convergence_cut_off_values = cut_off_values[relative_vector <= conv_criterion]
            return cut_off_values, save_convergence_cut_off_values
        
        elif len(relative_vector) == len(cut_off_values) - 1:
            # Handle case where lengths differ by 1
            cut_off_values_new = cut_off_values[:-1]
            # Use vectorized operation to filter values
            save_convergence_cut_off_values = cut_off_values_new[relative_vector <= conv_criterion]
            return cut_off_values_new,save_convergence_cut_off_values
        
        else:
            raise ValueError("`relative_vector` and `cut_off_values` must have the same length or `cut_off_values` must be one element longer than `relative_vector`.")





