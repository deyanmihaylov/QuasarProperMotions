import argparse
import cpnest

import Config as C
import Utils as U
import AstrometricData as AD
import Sampler as S
import PostProcessing as PP
import Plotting as P

# will be made obsolete
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description = "Quasar proper motions code")
    parser.add_argument(
        "parameter_file",
        metavar = "Parameter file",
        type = str,
        help = ".par file"
    )
    args = parser.parse_args()

    params = C.set_params(args.parameter_file)

    U.assert_config_params(params)

    C.check_output_dir(params['General']['output_dir'])

    C.record_config_params(params)

    data = AD.AstrometricDataframe()

    AD.load_astrometric_data(
        data,
        Lmax = params['Data']['Lmax'],
        N_obj = params['Data']['N_obj'],
        positions = params['Data']['positions'],
        positions_method = params['Data']['positions_method'],
        positions_seed = params['Data']['positions_seed'],
        bunch_size_polar = params['Data']['bunch_size_polar'],
        bunch_size_azimuthal = params['Data']['bunch_size_azimuthal'],
        proper_motions = params['Data']['proper_motions'],
        proper_motions_method = params['Data']['proper_motions_method'],
        proper_motions_seed = params['Data']['proper_motions_seed'],
        dipole = params['Data']['dipole'],
        multipole = params['Data']['multipole'],
        proper_motion_errors = params['Data']['proper_motion_errors'],
        proper_motion_errors_method = params['Data']['proper_motion_errors_method'],
        proper_motion_errors_std = params['Data']['proper_motion_errors_std'],
        proper_motion_errors_corr = params['Data']['proper_motion_errors_corr'],
        proper_motion_noise = params['Data']['proper_motion_noise'],
        proper_motion_noise_seed = params['Data']['proper_motion_noise_seed'],
        basis = params['Data']['basis']
    )

    astrometric_model = S.model(
                            data,
                            logL_method = params['MCMC']['logL_method'],
                            prior_bounds = params['MCMC']['prior_bounds']
                        )

    # nest = cpnest.CPNest(
    #             astrometric_model,
    #             output = params['General']['output_dir'],
    #             nthreads = params['MCMC']['nthreads'],
    #             nlive = params['MCMC']['nlive'],
    #             maxmcmc = params['MCMC']['maxmcmc'],
    #             resume = False,
    #             verbose = params['General']['verbose']
    #         )

    # nest.run()

    # nest.get_nested_samples(filename='nested_samples.dat')

    # nest.get_posterior_samples(filename='posterior.dat')

    # TO DO: Rewrite this with passing the samples instead of writing and reading a file. In March 2020 there is a bug in CPnest.
    A_limit = PP.post_process_results(
                    posterior_file = os.path.join(params['General']['output_dir'], 'type23_post.dat'),
                    which_basis = astrometric_model.which_basis,
                    Lmax = 4,
                    L = astrometric_model.overlap_matrix_Cholesky,
                    pol = params['Post_processing']['pol'],
                    limit = params['Post_processing']['limit']
                )

    U.export_data(
        data,
        A_limit,
        output = params['General']['output_dir']
    )

    if params['General']['plotting'] == True:
        P.plot(
            data,
            output = params['General']['output_dir']
        )

if __name__ == '__main__':
    main()
