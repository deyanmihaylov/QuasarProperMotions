import numpy as np
import sys
import itertools
from scipy.stats import truncnorm
from scipy.linalg import cholesky
import pandas as pd

import CoordinateTransformations as CT
import VectorSphericalHarmonics as VSH
import Utils as U
import Model as M


class AstrometricDataframe:
    def __init__(self):
        """
	    Initialise Class
        """
        self.N_obj = 0
        self.Lmax = None

        self.positions = np.array([])
        self.positions_Cartesian = np.array([])

        self.proper_motions = np.array([])

        self.inv_proper_motion_error_matrix = np.array([])

        self.basis = dict()
        self.which_basis = None

        self.names = dict()

        self.overlap_matrix = np.array([])
        self.overlap_matrix_Cholesky = np.array([])

    def generate_names(self):
        self.names = {
            (l, m, Q): f"Y^{Q}_{l},{m}"
            for l in range(1, self.Lmax+1)
            for m in range(-l, l+1) for Q in ['E', 'B']
        }

    def generate_positions(
            self,
            method: str,
            bunch_size_polar: float,
            bunch_size_azimuthal: float,
            random_seed: int
        ):
        """
        Generate random positions

        INPUTS
        ------
        method: string
            switches between a uniform distribution and a bunched (biased) distribution

        bunch_size_polar: float
            controls the distribution in the polar direction; 0. activates the uniform regime, while a small number (e.g. 0.1) is severely non-uniform

        bunch_size_azimuthal: float
            controls the distribution in the azimuthal direction; 0. activates the uniform regime, while a small number (e.g. 0.1) is severely non-uniform

        seed: int
            random seed
        """
        if random_seed > 0: np.random.seed(random_seed)

        if method == "uniform" or (method == "bunched" and bunch_size_polar==0.):
            dec = 0.5*np.pi - np.arccos(np.random.uniform(-1, 1, size=self.N_obj))
        elif method == "bunched" and bunch_size_polar > 0.:
            dec = 0.5*np.pi - np.arccos(truncnorm.rvs(-1./bunch_size_polar, 1./bunch_size_polar, scale=bunch_size_polar, size=self.N_obj))

        if method == "uniform" or (method == "bunched" and bunch_size_azimuthal==0.):
            ra = 2 * np.pi * np.random.uniform(0, 1, size=self.N_obj)
        elif method == "bunched" and bunch_size_azimuthal > 0.:
            ra = 2 * np.pi * (truncnorm.rvs(-0.5/bunch_size_azimuthal, 0.5/bunch_size_azimuthal, scale=bunch_size_azimuthal, size=self.N_obj)+0.5)

        self.positions = np.array(list(zip(ra, dec)))

    def load_Gaia_positions(
            self,
            dataset: pd.DataFrame
        ):
        """
        Load the positions from Gaia file
        """
        self.positions = dataset[['ra', 'dec']].values
        self.positions = U.deg_to_rad(self.positions)

    def load_TD_positions(
            self,
            dataset: pd.DataFrame
        ):
        """
        Load the positions from Truebenbach-Darling file
        """
        hours = 360. / 24.
        mins = hours / 60.
        secs = mins / 60.

        deg = 1.
        arcmin = deg / 60.
        arcsec = arcmin / 60.

        RAh = dataset['RAh'].values
        RAm = dataset['RAm'].values
        RAs = dataset['RAs'].values

        DEd = dataset['DEd'].values
        DEm = dataset['DEm'].values
        DEs = dataset['DEs'].values

        ra = RAh*hours + RAm*mins + RAs*secs
        dec = DEd*deg + DEm*arcmin + DEs*arcsec

        self.positions = np.transpose([ra, dec])
        self.positions = U.deg_to_rad(self.positions)

    def generate_proper_motions(
            self,
            method: str,
            dipole: float,
            multipole: list,
            random_seed: int
        ):

        if random_seed > 0: np.random.seed(random_seed)

        if method == "zero":
            self.proper_motions = np.zeros((self.N_obj, 2))
        elif method == "dipole":
            almQ = {
                (l, m, Q): 0.
                for l in range(1, self.Lmax+1)
                for m in range(-l, l+1) for Q in ['E', 'B']
            }

            almQ[(1, 0, 'E')] = dipole

            self.proper_motions = M.generate_model(almQ, self.basis)
        elif method == "multipole":
            almQ = {
                (l, m, Q): np.random.normal(0, multipole[l])
                for l in range(1, self.Lmax+1)
                for m in range(-l, l+1) for Q in ['E', 'B']
            }

            self.proper_motions = M.generate_model(almQ, self.basis)

    def load_Gaia_proper_motions(
            self,
            dataset: pd.DataFrame
        ):
        """
        Load the proper motions from Gaia file
        """
        self.proper_motions = dataset[['pmra', 'pmdec']].values

    def load_TD_proper_motions(
            self,
            dataset: pd.DataFrame
        ):
        """
        Load the proper motions from Truebenbach-Darling file
        """

        self.proper_motions = dataset[['pmRA', 'pmDE']].values

    def generate_proper_motion_errors(
            self,
            method: str,
            std: float,
            corr: float
        ):
        if method == "flat":
            scale = std
        elif method == "adaptive":
            scale = std * np.abs(self.proper_motions)

        proper_motion_errors = scale * np.ones(self.proper_motions.shape)

        # Scale the pm_ra_err by sin(theta) = cos(dec)
        proper_motion_errors[:,0] = proper_motion_errors[:,0] / np.cos(self.positions[:,1])

        proper_motion_errors_corr = corr * np.ones(self.N_obj)

        covariance = U.covariant_matrix(
            proper_motion_errors,
            proper_motion_errors_corr
        )

        self.inv_proper_motion_error_matrix = np.linalg.inv(covariance)

    def load_Gaia_proper_motion_errors(
            self,
            dataset: pd.DataFrame
        ):
        """
        Load the proper motion errors from Gaia file
        """
        proper_motions_errors = dataset[['pmra_error', 'pmdec_error']].values
        proper_motions_errors[:,0] = proper_motions_errors[:,0] / np.cos(self.positions[:,1])

        proper_motions_err_corr = dataset[['pmra_pmdec_corr']].values

        covariance = U.covariant_matrix(
            proper_motions_errors,
            proper_motions_err_corr
        )

        self.inv_proper_motion_error_matrix = np.linalg.inv(covariance)

    def load_TD_proper_motion_errors(
            self,
            dataset: pd.DataFrame
        ):
        """
        Load the proper motion errors from Truebenbach-Darling file
        TO DO: Use chi2 statistics for correlation
        """
        proper_motions_errors = dataset[['e_pmRA', 'e_pmDE']].values
        proper_motions_errors[:,0] = proper_motions_errors[:,0] / np.cos(self.positions[:,1])

        proper_motions_err_corr = np.zeros(self.N_obj)

        covariance = U.covariant_matrix(
            proper_motions_errors,
            proper_motions_err_corr
        )

        self.inv_proper_motion_error_matrix = np.linalg.inv(covariance)

    def add_proper_motion_noise(
            self,
            std: float,
            random_seed: int
        ):
        print("yes", std, random_seed)

        if random_seed > 0: np.random.seed(random_seed)

        proper_motion_noise = np.random.normal(
            loc=0.,
            scale=std,
            size=self.proper_motions.shape
        )
        
        self.proper_motions += proper_motion_noise

    def generate_VSHs(self):
        """
        Precompute VSH functions at QSO locations
        """
        def VSHs(l, m, Q):
            if Q == "E":
                return CT.Cartesian_to_geographic_vector(
                    self.positions_Cartesian,
                    VSH.RealVectorSphericalHarmonicE(
                        l,
                        m,
                        self.positions_Cartesian
                    )
                )
            elif Q == "B":
                return CT.Cartesian_to_geographic_vector(
                    self.positions_Cartesian,
                    VSH.RealVectorSphericalHarmonicB(
                        l,
                        m,
                        self.positions_Cartesian
                    )
                )

        self.basis = {
            (l, m, Q): VSHs(l, m, Q)
            for l in range(1, self.Lmax+1)
            for m in range(-l, l+1) for Q in ['E', 'B']
        }

        self.which_basis = "vsh"
    
    def remove_outliers(
            self,
            R_threshold: float
        ):
        """
        Remove outliers from dataset

        Define the dimensionless proper motion for each object as
        R = proper_motions^T . inverse_error_matrix . proper_motions

        We will remove outliers with R greater than a given threshold

        INPUTS
        ------
        R_threshold: float
            remove outliers with R>R_threshold
        """

        if R_threshold is not None:
            R = np.sqrt(np.einsum(
                '...i, ...ij, ...j -> ...',
                self.proper_motions,
                self.inv_proper_motion_error_matrix,
                self.proper_motions)
            )

            remove_mask = np.asarray(R > R_threshold)
            remove_indices = remove_mask.nonzero()[0]

            self.positions = np.delete(
                self.positions,
                remove_indices,
                axis=0
            )

            self.positions_Cartesian = np.delete(
                self.positions_Cartesian,
                remove_indices,
                axis=0
            )

            self.proper_motions = np.delete(
                self.proper_motions,
                remove_indices,
                axis=0
            )

            self.inv_proper_motion_error_matrix = np.delete(
                self.inv_proper_motion_error_matrix,
                remove_indices,
                axis=0
            )

            N_removed_outliers = remove_indices.shape[0]

            if N_removed_outliers == 1:
                print(f"Removed 1 outlier.")
            else:
                print(f"Removed {N_removed_outliers} outliers.")

    def compute_overlap_matrix(
            self,
            weighted_overlaps=True
        ):
        """
        Calculate the overlap matrix (and its Cholesky decomposition) 
        between VSH basis functions

        weighted_overlaps: bool
            whether or not to use the error weighted overlap sums
        """
        prefactor = 4. * np.pi / self.N_obj

        overlap_matrix_size = 2 * self.Lmax * (self.Lmax+2)

        self.overlap_matrix = np.zeros(
            (overlap_matrix_size, overlap_matrix_size)
        )

        if weighted_overlaps == False:
            metric = np.zeros((self.N_obj, 2, 2))
            metric[:, 0, 0] = np.cos(self.positions[:,1].copy())**2.
            metric[:, 1, 1] = 1

        basis_values = np.array(list(self.basis.values()))

        for ((i, vsh_i), (j, vsh_j)) in itertools.product(enumerate(basis_values), repeat=2):
            if weighted_overlaps == True:
                self.overlap_matrix[i,j] = prefactor * np.einsum(
                    "...i,...ij,...j->...",
                    vsh_i,
                    self.inv_proper_motion_error_matrix,
                    vsh_j
                ).sum()
            else:
                self.overlap_matrix[i,j] = prefactor * np.einsum(
                    "...i,...ij,...j->...",
                    vsh_i,
                    metric,
                    vsh_i
                ).sum()

        self.overlap_matrix = U.normalize_matrix(
            self.overlap_matrix,
            L=self.Lmax
        )

    def change_basis(
            self
        ):
        """
        Method to change from VSH basis to orthogonal basis
        """

        self.overlap_matrix_Cholesky = cholesky(self.overlap_matrix)

        invL = np.linalg.inv(self.overlap_matrix_Cholesky)

        vsh_basis_values = np.array(list(self.basis.values()))

        orthogonal_basis_values = np.einsum(
            "i...j,ik->k...j",
            vsh_basis_values,
            invL
        )

        self.basis = {
            key: orthogonal_basis_values[i]
            for i, key in enumerate(self.names)
        }

        self.which_basis = "orthogonal"

        self.compute_overlap_matrix()


def load_astrometric_data(
        ADf: AstrometricDataframe,
        params: dict
    ):
    ADf.Lmax = params['Lmax']

    ADf.generate_names()

    positions = params['positions']
    proper_motions = params['proper_motions']
    proper_motion_errors = params['proper_motion_errors']

    dataset_dict = {
        2: {"cat": "Gaia", "file_name": "data/type2.csv"},
        3: {"cat": "Gaia", "file_name": "data/type3.csv"},
        4: {"cat": "Gaia", "file_name": "data/type2and3.csv"},
        5: {"cat": "TD", "file_name": "data/TD6.dat"}
    }

    which_dataset = set(
        [
            positions,
            proper_motions,
            proper_motion_errors
        ]
        ).intersection(set(dataset_dict.keys()))

    if len(which_dataset) > 1:
        sys.exit("Conflicting datasets cannot be combined.")
    elif len(which_dataset) == 1:
        chosen_dataset = next(iter(which_dataset))

        if dataset_dict[chosen_dataset]['cat'] == "Gaia":
            dataset = import_Gaia_dataset(
                dataset_dict[chosen_dataset]['file_name']
            )
        elif dataset_dict[chosen_dataset]['cat'] == "TD":
            dataset = import_TD_dataset(
                dataset_dict[chosen_dataset]['file_name']
            )
    else:
        dataset = None

    if dataset is None:
        ADf.N_obj = params['N_obj']
    else:
        ADf.N_obj = dataset.shape[0]
    
    if positions == 1:
        ADf.generate_positions(
            method = params['positions_method'],
            bunch_size_polar = params['bunch_size_polar'],
            bunch_size_azimuthal = params['bunch_size_azimuthal'],
            random_seed = params['positions_seed']
        )
    elif positions in [2, 3, 4]:
        ADf.load_Gaia_positions(dataset)
    elif positions == 5:
        ADf.load_TD_positions(dataset)

    ADf.positions_Cartesian = CT.geographic_to_Cartesian_point(
        ADf.positions
    )

    if proper_motions == 1:
        ADf.generate_proper_motions(
            method = params['proper_motions_method'],
            dipole = params['dipole'],
            multipole = params['multipole'],
            random_seed = params['proper_motions_seed']
        )
    elif proper_motions in [2, 3, 4]:
        ADf.load_Gaia_proper_motions(dataset)
    elif proper_motions == 5:
        ADf.load_TD_proper_motions(dataset)

    if proper_motion_errors == 1:
        ADf.generate_proper_motion_errors(
            method = params['proper_motion_errors_method'],
            std = params['proper_motion_errors_std'],
            corr = params['proper_motion_errors_corr']
        )
    elif proper_motion_errors in [2, 3, 4]:
        ADf.load_Gaia_proper_motion_errors(dataset)
    elif proper_motion_errors == 5:
        ADf.load_TD_proper_motion_errors(dataset)

    if 'proper_motion_noise' in params:
        if 'proper_motion_noise_seed' in params:
            noise_seed = params['proper_motion_noise_seed']
        else:
            noise_seed = 0

        ADf.add_proper_motion_noise(
            std=params['proper_motion_noise'],
            random_seed=noise_seed
        )

    if 'dimensionless_proper_motion_threshold' in params:
        R_threshold = params['dimensionless_proper_motion_threshold']
    else:
        R_threshold = None

    ADf.remove_outliers(
        R_threshold
    )

    ADf.generate_VSHs()

    ADf.compute_overlap_matrix()

    if params['basis'] == "orthogonal":
        ADf.change_basis()
        ADf.compute_overlap_matrix()

def import_Gaia_dataset(
        path    
    ):
    """
    Import Gaia dataset
    """
    dataset = pd.read_csv(
        path,
        sep=',',
        header='infer',
        squeeze=False,
        mangle_dupe_cols=True,
        engine='c',
        skipinitialspace=False,
        skipfooter=0,
        keep_default_na=True,
        na_filter=True,
        verbose=False,
        skip_blank_lines=True,
        parse_dates=False,
        infer_datetime_format=False,
        keep_date_col=False,
        dayfirst=False,
        iterator=False,
        decimal=b'.',
        doublequote=True,
        error_bad_lines=True,
        warn_bad_lines=True,
        delim_whitespace=False,
        low_memory=False,
        memory_map=False
    )

    dropna_columns = [
        'ra',
        'dec',
        'pmra',
        'pmdec',
        'pmra_error',
        'pmdec_error',
        'pmra_pmdec_corr'
    ]

    dataset.dropna(
        axis=0,
        how='any',
        thresh=None,
        subset=dropna_columns,
        inplace=True
    )

    return dataset

def import_TD_dataset(
        path: str
    ):
    """
    Import TD dataset
    """

    col_names = [
        'Name',
        'RAh',
        'RAm',
        'RAs',
        'e_RAs',
        'DEd',
        'DEm',
        'DEs',
        'e_DEs',
        'pmRA',
        'e_pmRA',
        'o_pmRA',
        'chi2a',
        'pmDE',
        'e_pmDE',
        'o_pmDE',
        'chi2d',
        'Length',
        'MJD',
        'Flag',
        'z',
        'f_z',
        'r_z'
    ]

    dataset = pd.read_fwf(
        path,
        colspecs='infer',
        names=col_names,
        widths=None,
        comment = '#',
        infer_nrows=500
    )

    dropna_columns = [
        'RAh',
        'RAm',
        'RAs',
        'e_RAs',
        'DEd',
        'DEm',
        'DEs',
        'e_DEs',
        'pmRA',
        'e_pmRA',
        'o_pmRA',
        'chi2a',
        'pmDE',
        'e_pmDE',
        'o_pmDE',
        'chi2d'
    ]

    dataset.dropna(
        axis=0,
        how='any',
        thresh=None,
        subset=dropna_columns,
        inplace=True
    )

    return dataset
