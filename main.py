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
    parser = argparse.ArgumentParser(description="Quasar proper motions code")
    parser.add_argument("parameter_file", metavar="Parameter file", type=str, help=".par file")
    args = parser.parse_args()

    params = C.set_params(args.parameter_file)

    U.assert_config_params(params)

    C.check_output_dir(params['General']['output_dir'])

    C.record_config_params(params)

    data = AD.AstrometricDataframe()

    AD.load_astrometric_data(data,
                             Lmax = params['Analysis']['Lmax'],
                             N_obj = params['Analysis']['N_obj'],
                             positions = params['Analysis']['positions'],
                             positions_method = params['Analysis']['positions_method'],
                             positions_seed = params['Analysis']['positions_seed'],
                             bunch_size_polar = params['Analysis']['bunch_size_polar'],
                             bunch_size_azimuthal = params['Analysis']['bunch_size_azimuthal'],
                             proper_motions = params['Analysis']['proper_motions'],
                             proper_motions_method = params['Analysis']['proper_motions_method'],
                             proper_motions_seed = params['Analysis']['proper_motions_seed'],
                             dipole = params['Analysis']['dipole'],
                             multipole = params['Analysis']['multipole'],
                             proper_motion_errors = params['Analysis']['proper_motion_errors'],
                             proper_motion_errors_method = params['Analysis']['proper_motion_errors_method'],
                             proper_motion_errors_std = params['Analysis']['proper_motion_errors_std'],
                             proper_motion_errors_corr_method = params['Analysis']['proper_motion_errors_corr_method'],
                             proper_motion_noise = params['Analysis']['proper_motion_noise'],
                             proper_motion_noise_seed = params['Analysis']['proper_motion_noise_seed'],
                             basis = params['Analysis']['basis']
                            )

    astrometric_model = S.model(data,
                                logL_method = params['MCMC']['logL_method'],
                                prior_bounds = params['MCMC']['prior_bounds']
                               )

    nest = cpnest.CPNest(astrometric_model,
                         output = params['General']['output_dir'],
                         nthreads = params['MCMC']['nthreads'],
                         nlive = params['MCMC']['nlive'],
                         maxmcmc = params['MCMC']['maxmcmc'],
                         resume=False,
                         verbose=2
                        )

    nest.run()

    nest.get_nested_samples(filename='nested_samples.dat')

    nest.get_posterior_samples(filename='posterior.dat')

    # TO DO: Rewrite this with passing the samples instead of writing and reading a file. In March 2020 there is a bug in CPnest.
    A_limit = PP.post_process_results(posterior_file = os.path.join(params['General']['output_dir'], 'posterior.dat'),
                                      which_basis = astrometric_model.which_basis,
                                      Lmax = params['Analysis']['Lmax'],
                                      L = astrometric_model.overlap_matrix_Cholesky,
                                      pol = params['Post_processing']['pol'],
                                      limit = params['Post_processing']['limit']
                                     )

    np.savetxt(os.path.join(params['General']['output_dir'], "limit.dat"), np.array([A_limit]))

    U.export_data(data,
                  output = params['General']['output_dir'])

    if params['General']['plotting'] == True:
        P.plot(data,
               output = params['General']['output_dir']
              )

if __name__ == '__main__':
    main()
