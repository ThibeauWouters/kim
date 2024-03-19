from abc import ABC, abstractmethod

from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped
from dataclasses import field
from typing import Callable, Union
import jax
import jax.numpy as jnp
import numpy as np
import json

from flowMC.sampler.Sampler import Sampler
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Gaussian_random_walk import GaussianRandomWalk
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.utils.EvolutionaryOptimizer import EvolutionaryOptimizer
from flowMC.nfmodel.base import Distribution

from jimgw.base import LikelihoodBase
from jimgw.prior import Prior, Uniform, Composite
from jimgw.jim import default_hyperparameters

class Kim(object):
    """
    Small class to interface with flowMC for the KN likelihood.
    """
    
    def __init__(self, 
                 likelihood: LikelihoodBase, 
                 prior: Prior, **kwargs):
        self.Likelihood = likelihood
        self.Prior = prior

        # Set and override any given hyperparameters, and save as attribute
        self.hyperparameters = default_hyperparameters
        hyperparameter_names = list(self.hyperparameters.keys())
        
        for key, value in kwargs.items():
            if key in hyperparameter_names:
                self.hyperparameters[key] = value
        
        for key, value in self.hyperparameters.items():
            setattr(self, key, value)

        rng_key_set = initialize_rng_keys(self.hyperparameters["n_chains"], seed=self.hyperparameters["seed"])
        local_sampler_arg = kwargs.get("local_sampler_arg", {})

        # # Set the local sampler
        # local_sampler = GaussianRandomWalk(
        #     self.posterior, False, local_sampler_arg
        # )  # Remember to add routine to find automated mass matrix
        
        local_sampler = MALA(
                self.posterior, True, local_sampler_arg
            )  # Remember to add routine to find automated mass matrix

        model = MaskedCouplingRQSpline(
            self.Prior.n_dim, self.num_layers, self.hidden_size, self.num_bins, rng_key_set[-1]
        )
        global_sampler = None

        self.Sampler = Sampler(
            self.Prior.n_dim,
            rng_key_set,
            None,  # type: ignore
            local_sampler,
            model,
            global_sampler=global_sampler,
            **kwargs,
        )
        
    def posterior(self, params: Float[Array, " n_dim"], data: dict):
        prior_params = self.Prior.add_name(params.T)
        prior = self.Prior.log_prob(prior_params)
        return (
            self.Likelihood.evaluate(self.Prior.transform(prior_params), data) + prior
        )
        
    def sample(self, key: PRNGKeyArray, initial_guess: Array = jnp.array([])):
        if initial_guess.size == 0:
            initial_guess_named = self.Prior.sample(key, self.Sampler.n_chains)
            initial_guess = jnp.stack([i for i in initial_guess_named.values()]).T
            
        self.initial_guess = initial_guess
        self.Sampler.sample(initial_guess, None)  # type: ignore
        
        
    def get_samples(self, training: bool = False) -> dict:
        """
        Get the samples from the sampler

        Parameters
        ----------
        training : bool, optional
            Whether to get the training samples or the production samples, by default False

        Returns
        -------
        dict
            Dictionary of samples

        """
        if training:
            chains = self.Sampler.get_sampler_state(training=True)["chains"]
        else:
            chains = self.Sampler.get_sampler_state(training=False)["chains"]

        chains = self.Prior.transform(self.Prior.add_name(chains.transpose(2, 0, 1)))
        return chains