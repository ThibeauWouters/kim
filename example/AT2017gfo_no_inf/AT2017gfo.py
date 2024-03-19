import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import inspect
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
# matlotlib settings
mpl_params = {"axes.grid": True,
        "text.usetex" : False, # TODO enable latex, but this breaks if filters have underscore
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}
plt.rcParams.update(mpl_params)

import nmma as nmma
from nmma.em.io import loadEvent
from nmma.em.model import SVDLightCurveModel
import nmma.em.model_parameters as model_parameters

# flowMC imports
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.sampler.Gaussian_random_walk import GaussianRandomWalk
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

from jimgw.prior import Uniform, Composite

from kim.likelihood import OpticalLightCurve
from kim.kim import Kim

# jax imports
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", False)
import jax.numpy as jnp

import corner

print("Checking if CUDA is found:")
print(jax.devices())

### LOAD THE DATA ###

data_file = "../../data/AT2017gfo_corrected_no_inf.dat"
trigger_time = 57982.5285236896
tmin, tmax = 0.05, 14
data = loadEvent(data_file)
filters = list(data.keys())
print("Filters: ", filters)
sample_times = jnp.linspace(tmin, tmax, 1_000)

### PRIORS ###

log10_mej_dyn_prior = Uniform(-3.0, -1.7, naming=["log10_mej_dyn"])
vej_dyn_prior = Uniform(0.12, 0.25, naming=["vej_dyn"])
Yedyn_prior = Uniform(0.15, 0.3, naming=["Yedyn"])
log10_mej_wind_prior = Uniform(-2.0, -0.89, naming=["log10_mej_wind"])
vej_wind_prior = Uniform(0.03, 0.15, naming=["vej_wind"])
inclination_EM_prior = Uniform(0., np.pi/2., naming=["inclination_EM"])

prior_list = [log10_mej_dyn_prior, 
              vej_dyn_prior, 
              Yedyn_prior, 
              log10_mej_wind_prior, 
              vej_wind_prior, 
              inclination_EM_prior]

n_dim = len(prior_list)
prior_range = [[prior.xmin, prior.xmax] for prior in prior_list]
parameter_naming = [prior.naming for prior in prior_list]
composite_prior = Composite(prior_list)

### LIKELIHOOD ###

MODEL_FUNCTIONS = {
    k: v for k, v in model_parameters.__dict__.items() if inspect.isfunction(v)
}
MODEL_NAME = "Bu2022Ye"
model_function = MODEL_FUNCTIONS[MODEL_NAME]

svd_path = "/home/urash/twouters/flax_models"
lc_model = SVDLightCurveModel(
                MODEL_NAME,
                sample_times,
                svd_path=svd_path,
                parameter_conversion=None,
                mag_ncoeff=10,
                lbol_ncoeff=None,
                interpolation_type="flax",
                model_parameters=None,
                filters=filters,
                local_only=True
)

fixed_params = {'luminosity_distance': 44.0, 
                 'timeshift': 0.0}

likelihood = OpticalLightCurve(lc_model,
                               filters,
                               data,
                               trigger_time,
                               fixed_params=fixed_params,
)

### LOCAL SAMPLER ARG ###

eps = 1e-5
mass_matrix = jnp.eye(n_dim)
# TODO tune it here
# mass_matrix = mass_matrix.at[0,0].set(1e-5)
# mass_matrix = mass_matrix.at[1,1].set(1e-4)
# mass_matrix = mass_matrix.at[2,2].set(1e-3)
# mass_matrix = mass_matrix.at[3,3].set(1e-3)
# mass_matrix = mass_matrix.at[7,7].set(1e-5)
# mass_matrix = mass_matrix.at[11,11].set(1e-2)
# mass_matrix = mass_matrix.at[12,12].set(1e-2)
local_sampler_arg = {"step_size": mass_matrix * eps}

outdir_name = "./outdir/"
kim = Kim(likelihood,
          composite_prior,
          n_loop_training=5,
          n_loop_production=5,
          n_local_steps=20,
          n_global_steps=20,
          n_chains=100,
          n_epochs=10,
          learning_rate=0.001,
          max_samples=50000,
          momentum=0.9,
          batch_size=50000,
          use_global=True,
          keep_quantile=0.0,
          train_thinning=10,
          output_thinning=40,
          local_sampler_arg=local_sampler_arg,
          outdir_name=outdir_name
          )

kim.sample(jax.random.PRNGKey(42))

samples = kim.get_samples()
np.savez("samples.npz", samples=samples)

print("DONE")