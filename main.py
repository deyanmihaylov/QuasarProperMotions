#!/usr/bin/env python
import argparse
import cpnest
import bilby

import Config as C
import Utils as U
import AstrometricData as AD
import sampler
import PostProcess as PP
import Plotting as P

# will be made obsolete
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(
        description = "Quasar proper motions code",
    )

    parser.add_argument(
        "parameter_file",
        metavar = "Parameter file",
        type = str,
        help = ".par file",
    )
    args = parser.parse_args()
    params = C.set_params(args.parameter_file)
    C.prepare_output_dir(params["General"]["output_dir"])
    C.record_config_params(params)
    
    data = AD.AstrometricDataframe()
    
    AD.load_astrometric_data(
        data,
        params = params["Data"],
    )

    bilby.core.utils.setup_logger(
        outdir=params['General']['output_dir'],
        label="bilby_output",
    )

    names_ordered = [
        data.almQ_names[lmQ] for lmQ in data.lmQ_ordered
    ]

    if params['MCMC']["logL_method"] == "goodandbad":
        names_ordered.extend(["log10_beta", "log10_gamma"])

    priors = {
        par: bilby.core.prior.Uniform(-0.2, 0.2, par) for par in names_ordered
    }

    if params['MCMC']["logL_method"] == "goodandbad":
        priors["log10_beta"] = bilby.core.prior.Uniform(-1.78, -1.20, "log10_beta")
        priors["log10_gamma"] = bilby.core.prior.Uniform(-0.08, 0.52, "log10_gamma")

    likelihood = sampler.QuasarProperMotionLikelihood(
        data,
        params = params['MCMC'],
    )

    result = bilby.run_sampler(
        outdir=params['General']['output_dir'],
        label="bilby_output",
        resume=False,
        plot=True,
        likelihood=likelihood,
        priors=priors,
        sampler="nessai",
        # injection_parameters={'x': 0.0, 'y': 0.0},
        analytic_priors=False,
        seed=1234,
        nlive=1024,
    )

    # nested_samples = nest.get_nested_samples(filename=None)
    # np.savetxt(
    #     os.path.join(params['General']['output_dir'], 'nested_samples.dat'),
    #     nested_samples.ravel(),
    #     header=' '.join(nested_samples.dtype.names),
    #     newline='\n',
    #     delimiter=' ',
    # )

    # posterior_samples = nest.get_posterior_samples(filename=None)
    # np.savetxt(
    #     os.path.join(params['General']['output_dir'], 'posterior_samples.dat'),
    #     posterior_samples.ravel(),
    #     header=' '.join(posterior_samples.dtype.names),
    #     newline='\n',
    #     delimiter=' ',
    # )

    # nest.plot()

    # PP.post_process_results(
    #     posterior_file = os.path.join(params['General']['output_dir'], 'posterior_samples.dat'),
    #     Lmax = params['Data']['Lmax'],
    #     outdir = params['General']['output_dir'],
    # )

    # U.export_data(
    #     data,
    #     A_limit,
    #     output = params['General']['output_dir']
    # )

    # if params['General']['plotting'] == True:
    #     P.plot(
    #         data,
    #         output = params['General']['output_dir']
    #     )

if __name__ == '__main__':
    main()
