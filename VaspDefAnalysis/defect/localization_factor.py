from VaspDefAnalysis.defect.defect_analisys import DefectAnalisys
from VaspDefAnalysis.defect.tool_defect import get_ion_orbital_total_weight_neighbors_to_defects,get_ipr_values

class LocalizedStates:
    
    """
    Class for calculating localization factors based on different approaches such as: Inderect method using Ion Orbital Weights (IOWs) and direct using 
    Inverse Participation Ratio (IPR).
    """

    @staticmethod
    def get_localization_factor_using_IOWs(perfect_structure_path:str,
                                           defect_structure_path:str,
                                           defect_vasprun_path:str,
                                           radius:float,
                                           tolerance=1e-3,
                                           norm:bool= False
                                           ):
        """
        Calculate the localization factor using Ion Orbital Weights (IOWs) for a defect.

        This method analyzes the ion orbital weights of neighbors around a defect in a material to assess the degree of 
        localization of the defect state.

        Parameters:
        ----------
        perfect_structure_file : str
            Path to the perfect structure.
        defect_structure_file : str
            Path to the defect structure without relaxing.
        radius : float
            The radius within which atoms are considered neighbors.
        tolerance : float
            Tolerance for identifying defect. Default is 1e-3.

        Note: The defect_structure_path has to be without relaxing otherwise the result can be wrong.
        """

        ion_neighbor_indeces_to_defects = DefectAnalisys.get_ion_neighbor_indices_to_defects(perfect_structure_file=perfect_structure_path,
                                                                                             defect_structure_file=defect_structure_path,
                                                                                             radius=radius,
                                                                                             tolerance=tolerance)
        ion_orbital_total_weight_neighbors_to_defects = get_ion_orbital_total_weight_neighbors_to_defects(vasprun_path=defect_vasprun_path,
                                                                                                          ion_neighbor_indeces_to_defects=ion_neighbor_indeces_to_defects,
                                                                                                          norm=norm)
        
        return ion_orbital_total_weight_neighbors_to_defects


    
    @staticmethod
    def get_localization_factor_using_IPR(WAVECAR_path:str,
                                          lsorbit:bool=False, 
                                          lgamma:bool=False,
                                          gamma_half:str='x', 
                                          omp_num_threads:int=1,
                                          norm:bool= True,
                                          **settings):
        """
        Calculate the localization factor using the Inverse Participation Ratio (IPR) for a defect.

        The IPR is a measure of the localization of electronic states in a material, and this method calculates the IPR 
        based on the wavefunctions provided in the WAVECAR file.

            Parameters:
        -----------
        WAVECAR_path : str
            The path to the WAVECAR file containing the wavefunction data.

        lsorbit : bool, optional, default=False
            If True, includes the orbital angular momentum in the wavefunction calculation.

        lgamma : bool, optional, default=False
            If True, includes the Gamma point in the calculations.

        gamma_half : str, optional, default='x'
            Specifies the Gamma half-point calculation ('x' is the default value).

        omp_num_threads : int, optional, default=1
            The number of OpenMP threads to use for parallel computation.

        settings:   
            gvec : the G-vectors correspond to the plane-wave coefficients
            Cg : the plane-wave coefficients. If None, read from WAVECAR
            ngrid : the FFT grid size
            norm : normalized Cg?
            kr_phase : whether or not to multiply the exp(ikr) phase
            r0 : shift of the kr-phase to get full wfc other than primitive cell

        Returns:
        --------    
        dict
            A dictionary with the IPR values for each spin, k-point, and band {spin: {kpoint index: [band1 band2 ...]}}.
            Where each band has its IPR value.
        
        Note:
        -----
        This function uses the `vaspwfc` class from the `VaspBandUnfolding` library (https://github.com/QijingZheng/VaspBandUnfolding.git) to read the WAVECAR file and extract wavefunction data.
        """

        r""" 
        IPR is defined:

                        \sum_n |\phi_j(n)|^4 
        IPR(\phi_j) = -------------------------
                      |\sum_n |\phi_j(n)|^2||^2
        """
        
        return get_ipr_values(WAVECAR_path=WAVECAR_path,
                              lsorbit=lsorbit,
                              lgamma=lgamma,
                              gamma_half=gamma_half,
                              omp_num_threads=omp_num_threads,
                              norm=norm,
                              **settings)