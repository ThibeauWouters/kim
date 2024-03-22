import copy
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import truncnorm, norm
from scipy.interpolate import interp1d
import inspect 
from nmma.em.io import loadEvent
from nmma.em.utils import calc_lc_flax, calc_lc, getFilteredMag
from nmma.em.model import SVDLightCurveModel
import nmma.em.model_parameters as model_parameters
from nmma.em.utils import dataProcess
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped
from abc import ABC, abstractmethod

from jimgw.base import LikelihoodBase

def truncated_gaussian(m_det, m_err, m_est, upper_lim, lower_lim = -9999.0):
    # TODO: move this to an utils?

    a, b = (lower_lim - m_est) / m_err, (upper_lim - m_est) / m_err
    logpdf = truncnorm.logpdf(m_det, a, b, loc=m_est, scale=m_err)
    # logpdf = norm.logpdf(m_est, loc=m_det, scale=m_err)

    return logpdf

class OpticalLightCurve(LikelihoodBase):
    
    # TODO: get the non-detections in here as well
    def __init__(self, 
                 light_curve_model: SVDLightCurveModel,
                 filters: list,
                 light_curve_data: dict,
                 trigger_time: float,
                 detection_limit: float = 9999.0,
                 fixed_params: dict = {},
                 error_budget: float = 1.0,
                 tmin: float = 0.0,
                 tmax: float = 14.0,
                 verbose: bool = False):
        """
        Initialize the OpticalLightCurve class, the light curve model should have flax interpolation type and calls the NMMA SVD model. 

        Args:
            light_curve_model (SVDLightCurveModel): _description_
            filters (list): _description_
            light_curve_data (dict): _description_
            trigger_time (float): _description_
            detection_limit (float, optional): _description_. Defaults to None.
            error_budget (float, optional): _description_. Defaults to 1.0.
            tmin (float, optional): _description_. Defaults to 0.0.
            tmax (float, optional): _description_. Defaults to 14.0.
            verbose (bool, optional): _description_. Defaults to False.
        """
        
        self.light_curve_model = light_curve_model
        self.fixed_params = fixed_params
        self.filters = filters
        self.detection_limit = detection_limit
        self.error_budget = error_budget
        self.tmin = tmin
        self.tmax = tmax
        self.verbose = verbose
        
        # Process the data
        processedData = dataProcess(
            light_curve_data, self.filters, trigger_time, self.tmin, self.tmax
        )
        self.light_curve_data = processedData
        self.sample_times = self.light_curve_model.sample_times
    
    def get_chisq_filt(self,
                       mag_app: jnp.array, 
                       data_time: jnp.array, 
                       data_mag: jnp.array, 
                       data_sigma: np.array,
                       timeshift: float = 0.0,
                       upper_lim: float = 9999.0, 
                       lower_lim: float = -9999.0
                    ):
        """
        Function taken from nmma/em/likelihood.py and adapted to this case here
        
        This is a piece of the log likelihood function, which is the sum of the chisquare for a single filter, to decompose the likelihood calculation.
        """
        
        data_sigma = jnp.sqrt(data_sigma ** 2 + self.error_budget ** 2)
        mag_est = jnp.interp(data_time, self.sample_times + timeshift, mag_app, left="extrapolate", right="extrapolate")
        
        minus_chisquare = jnp.sum(
            truncated_gaussian(
                data_mag,
                data_sigma,
                mag_est,
                upper_lim=upper_lim,
                lower_lim=lower_lim,
            )
        )
        return minus_chisquare
    
    def calc_lc(self,
                params: dict):
        
        # TODO: mag_ncoeff fetched?
        params.update(self.fixed_params)
        
        result_dict = {}
        if "KNtheta" not in params:
            params["KNtheta"] = (
                jnp.rad2deg(params["inclination_EM"])
            )
            
        params_array = jnp.array([params[key] for key in self.light_curve_model.model_parameters])
        _, _, mag_abs = calc_lc_flax(self.sample_times,
                                    params_array,
                                    svd_mag_model=self.light_curve_model.svd_mag_model,
                                    svd_lbol_model=None,
                                    mag_ncoeff=10,
                                    lbol_ncoeff=None,
                                    filters=self.filters)
    
        for filt in self.filters:
            mag_abs_filt = getFilteredMag(mag_abs, filt)
            mag_app_filt = mag_abs_filt + 5.0 * jnp.log10(params["luminosity_distance"] * 1e6 / 10.0)
            result_dict[filt] = mag_app_filt
            
        return result_dict

    def log_likelihood_chisq(self,
                             params: dict) -> float:
        """
        Function taken from nmma/em/likelihood.py and adapted to this case here

        Args:
            params (dict): Parameters for the light curve model

        Returns:
            float: Log likelihood of the chisquare
        """
        
        params.update(self.fixed_params)
        mag_app_dict = self.calc_lc(params)
        minus_chisquare_total = 0.0
        for filt in self.filters:
            # TODO: do we need the deepcopy here?
            data_time, data_mag, data_sigma  = self.light_curve_data[filt].T
            mag_est_filt = mag_app_dict[filt]
            chisq_filt = self.get_chisq_filt(mag_est_filt, data_time, data_mag, data_sigma, timeshift=params["timeshift"])
            minus_chisquare_total += chisq_filt

        log_prob = minus_chisquare_total

        return log_prob
    
    def evaluate(self,
                params: dict[str, Float], 
                data: dict,
                ) -> float:
        """
        Function taken from nmma/em/likelihood.py and adapted to this case here
        
        TODO: 
        - separate LC params from params?
        - add error budget
        - add timeshift
        - add luminosity distance
        - params: called with dict?
        - can remove data argument?
        - this is assuming all data are "finite" and the LC is finite. Not checking this here since breaks JAX jit
        """
        
        params.update(self.fixed_params)
        log_prob  = self.log_likelihood_chisq(params)
        # TODO add the non detections here
        log_prob += 0.0

        return log_prob
