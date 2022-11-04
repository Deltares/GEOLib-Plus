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
        """ """
        return slope * x + intercept

    @staticmethod
    def linear_regression(
        x: Union[float, np.array],
        y: Union[float, np.array],
    ) -> (dict, dict):
        """ """

        # initialise covariance matrix
        covariance_matrix = np.zeros((2, 2))

        popt, cov = curve_fit(ShansepUtils.__linear, x, y, method="lm")
        slope, intercept = popt
        # point statistics
        std_slope, std_intercept, covariance = (
            np.sqrt(cov[0, 0]),
            np.sqrt(cov[1, 1]),
            cov[0, 1],
        )
        rho = covariance / (std_slope * std_intercept)

        N = len(x)
        y_fit = ShansepUtils.__linear(x, slope, intercept)
        residuals = np.sum((y - y_fit) ** 2)

        def regression_function(x, alpha, quantile):
            Z = (
                (std_intercept**2)
                + (x**2 * std_slope**2)
                + 2 * rho * x * std_intercept * std_slope
                + (1 - alpha) * residuals / (N - 2)
            )

            tz = t(N - 2).ppf(quantile) * np.sqrt(Z)
            y = (a * x + b) + tz

        fit_params = {
            "slope": slope,
            "intercept": intercept,
            "std_slope": std_slope,
            "std_intercept": std_intercept,
        }

        other_params = {
            "n": n,
            "rho": rho,
            "covariance": covariance,
            "residuals": residuals,
        }

        return fit_params, other_params, regression_function
