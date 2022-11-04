import warnings
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel
from scipy.optimize import curve_fit

from geolib_plus.shm.prob_utils import ProbUtils


class RegressionUtils(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def __linear(x, slope, intercept):
        """ 
        Linear funciton as input for optimizing routine
        """
        return slope * x + intercept

    @staticmethod
    def linear_regression(
        x: Union[float, np.array],
        y: Union[float, np.array],
        bounds=([-np.inf,-np.inf],[np.inf,np.inf]) ,
        ) -> (dict, dict, tuple):
        """ 
        Do a linear regression between x and y. The inputs x and y are arrays. 
        You can specify bounds for the intercept and slope parameter. 
        In case of an unbounded problem the 'trf' method is used in stead of Levenberg Maquard. 
        """

        # initialise covariance matrix
        covariance_matrix = np.zeros((2, 2))

        if np.any((np.array(bounds[0]) > -np.inf) | (np.array(bounds[1]) < np.inf )): #bounded problem use trf
            method = 'trf'
        else: #unbounded problem use levenberg maquard
            method='lm'

        popt, cov = curve_fit(RegressionUtils.__linear, x, y, method=method, bounds=bounds)
        slope, intercept = popt
        # point statistics
        std_slope, std_intercept, covariance = (
            np.sqrt(cov[0, 0]),
            np.sqrt(cov[1, 1]),
            cov[0, 1],
        )
        rho = covariance / (std_slope * std_intercept)

        N = len(x)
        y_fit = RegressionUtils.__linear(x, slope, intercept)
        residuals = np.sum((y - y_fit) ** 2)

        fit_params = {'slope':slope,
                      'intercept':intercept,
                      'std_slope':std_slope,
                      'std_intercept':std_intercept,
                      }

        other_params = {'N':N,
                        'rho':rho,
                        'covariance':covariance,
                        'residuals':residuals
                        }

        def regression_function(x, alpha, quantile):
            Z = (std_intercept ** 2) \
                + (x ** 2 * std_slope ** 2) \
                + 2 * rho * x * std_intercept * std_slope \
                + (1 - alpha) * residuals / (N - 2)

            from scipy.stats import t
            tz = t(N - 2).ppf(quantile) * np.sqrt(Z)
            y = (slope * x + intercept) + tz

            return y

        return fit_params, other_params, regression_function
