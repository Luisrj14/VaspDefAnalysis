from typing import List, Dict
import xml.etree.ElementTree as ET
from vaspwfc import vaspwfc
import numpy as np

"""
Here there are some tools for defect class
"""

def get_ion_orbital_total_weight_neighbors_to_defects(
    vasprun_path: str,
    ion_neighbor_indeces_to_defects: List[int],
    norm: bool = False,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Extracts the total orbital weights of ions near a defect from a VASP `vasprun.xml`.

    Parameters:
    -----------
    vasprun_path : str
        Path to `vasprun.xml`.
    ion_neighbor_indeces_to_defects : list of int
        Indices of ions close to the defect to include in the calculation.
    norm : bool
        Whether to normalize weights.

    Returns:
    --------
    dict
        Nested dictionary: spin -> kpoint -> list of total band weights (one per band).
    """
    tree = ET.parse(vasprun_path)
    root = tree.getroot()

    weights_per_band = {}

    for spin in root.findall(".//set[@comment='spin1']") + root.findall(".//set[@comment='spin2']"):
        spin_key = spin.attrib['comment']
        weights_per_band[spin_key] = {}

        for kpoint_index, kpoint_set in enumerate(spin.findall("set")):
            kpoint_key = f"kpoint {kpoint_index + 1}"
            weights_per_band[spin_key][kpoint_key] = []

            for band in kpoint_set.findall("set"):
                band_weights = []
                for ion_index, r in enumerate(band.findall("r")):
                    if ion_index not in ion_neighbor_indeces_to_defects:
                        continue
                    orbital_weights = list(map(float, r.text.strip().split()))
                    total_weight = sum(orbital_weights)
                    band_weights.append(total_weight)

                # Store the sum of weights for this band at this k-point
                weights_per_band[spin_key][kpoint_key].append(sum(band_weights))
    
    # Normalize the weights if requested
    if norm:
        total_sum_per_spin = 0.0
        for spin_key, kpoint_data in weights_per_band.items():
            for kpoint_key, band_weights in kpoint_data.items():
                total_sum = sum(band_weights)
                total_sum_per_spin += total_sum
        for spin_key, kpoint_data in weights_per_band.items():
            for kpoint_key, band_weights in kpoint_data.items():    
                if total_sum_per_spin > 0:
                    normalized = [w / total_sum_per_spin for w in band_weights]
                    weights_per_band[spin_key][kpoint_key] = normalized
                else:
                    raise ValueError(
                        f"Zero total weight sum for {spin_key}, {kpoint_key}. Check neighbor indices or input data."
                    )

    return weights_per_band

def get_ipr_values(WAVECAR_path:str,
                   lsorbit:bool=False, 
                   lgamma:bool=False,
                   gamma_half:str='x', 
                   omp_num_threads:int=1,
                   **settings):
    
    '''
    Calculate the Inverse Participation Ratio (IPR) for each band, k-point, and spin from a WAVECAR file.

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
    '''
    r"""
    IPR definition:

                    \sum_n |\phi_j(n)|^4 
    IPR(\phi_j) = -------------------------
                  |\sum_n |\phi_j(n)|^2||^2

   The normalizatition is defined:

    \sum_{ijk} | \phi_{ijk} | ^ 2 = 1
    
    """    
    # Create an instance of vaspwfc
    read_vasp_wf = vaspwfc(fnm=WAVECAR_path,
                           lsorbit=lsorbit, 
                           lgamma=lgamma,
                           gamma_half=gamma_half, 
                           omp_num_threads=omp_num_threads
                           )

    # Initialize a dictionary to store IPR data
    ipr_values = {}

    # Loop over k-points, bands, and spins (assuming the WAVECAR contains this information)
    for spin in range(read_vasp_wf._nspin): 
        ipr_values[f'spin{spin}'] = {}

        for k_index in range(read_vasp_wf._nkpts):
            ipr_values[f'spin{spin}'][f"kpoint {k_index}"] = []

            for band_index in range(read_vasp_wf._nbands):
                # Get the real-space wavefunction
                phi_j = read_vasp_wf.wfc_r(ispin=spin+1, ikpt=k_index+1, iband=band_index+1,**settings)

                phi_j_abs = np.abs(phi_j)

                # Calculate the IPA for each band 
                ipr= np.sum(phi_j_abs**4) / np.sum(phi_j_abs**2)**2

                # Store the IPR in the dictionary
                ipr_values[f'spin{spin}'][f"kpoint {k_index}"].append(ipr)

    return ipr_values