#!/usr/bin/env python
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
    parser = argparse.ArgumentParser(
        description = "Quasar proper motions code"
    )

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
        params = params['Data']
    )

    astrometric_model = S.model(
        data,
        logL_method = params['MCMC']['logL_method'],
        prior_bounds = params['MCMC']['prior_bounds']
    )

    nest = cpnest.CPNest(
        astrometric_model,
        output = params['General']['output_dir'],
        nthreads = params['MCMC']['nthreads'],
        nlive = params['MCMC']['nlive'],
        maxmcmc = params['MCMC']['maxmcmc'],
        resume = True,
        verbose = params['General']['verbose']
    )

    nest.run()

    nested_samples = nest.get_nested_samples(filename=None)
    np.savetxt(
        os.path.join(params['General']['output_dir'], 'nested_samples.dat'),
        nested_samples.ravel(),
        header=' '.join(nested_samples.dtype.names),
        newline='\n',
        delimiter=' '
    )

    posterior_samples = nest.get_posterior_samples(filename=None)
    np.savetxt(
        os.path.join(params['General']['output_dir'], 'posterior_samples.dat'),
        posterior_samples.ravel(),
        header=' '.join(posterior_samples.dtype.names),
        newline='\n',
        delimiter=' '
    )

    A_limit = PP.post_process_results(
        posterior_file = os.path.join(params['General']['output_dir'], 'posterior_samples.dat'),
        which_basis = astrometric_model.which_basis,
        Lmax = params['Data']['Lmax'],
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
