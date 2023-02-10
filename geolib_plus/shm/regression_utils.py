# import packages
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
        Function for linear regression. Use as input within optimizing routine.
        """
        return slope * x + intercept

    @staticmethod
    def linear_regression(
        x: Union[float, np.array],
        y: Union[float, np.array],
        bounds=([-np.inf, -np.inf], [np.inf, np.inf]),
    ) -> (dict, dict, tuple):
        """
        Method for linear regression between x and y. The inputs x and y are arrays.
        The user may specify the bounds for the regression coefficients intercept and slope paramete.
        If the bounds are specified, the 'Trust Region Reflective algorithm' method is used in the optimizing routing,
        for unbounded problems the 'Levenberg-Marquardt' method is used.
        """

        # initialise covariance matrix
        covariance_matrix = np.zeros((2, 2))

        # For a bounded problem use trf method, for unbounded problem use levenberg maquard
        if np.any((np.array(bounds[0]) > -np.inf) or (np.array(bounds[1]) < np.inf)):
            method = "trf"
        else:
            method = "lm"

        popt, cov = curve_fit(
            RegressionUtils.__linear, x, y, method=method, bounds=bounds
        )
        slope, intercept = popt

        # get the standard deviation and  covariance of the residuals
        std_slope, std_intercept, covariance = (
            np.sqrt(cov[0, 0]),
            np.sqrt(cov[1, 1]),
            cov[0, 1],
        )
        rho = covariance / (std_slope * std_intercept)

        N = len(x)
        y_fit = RegressionUtils.__linear(x, slope, intercept)
        residuals = np.sum((y - y_fit) ** 2)

        fit_params = {
            "slope": slope,
            "intercept": intercept,
            "std_slope": std_slope,
            "std_intercept": std_intercept,
        }

        other_params = {
            "N": N,
            "rho": rho,
            "covariance": covariance,
            "residuals": residuals,
        }

        def regression_function(x, alpha=0, quantile=0.05):
            """
            After fitting the regression, the regression function y = slope * x + intercept is returned.
            A specific upper or lower limit of the regression line can be retrieved by specifying the:
            - quantile (default 5% lower limit: quantile = 0.05).
            The function accounts for statistical uncertainty (Student-t distribution) and the uncertainty
            of the fit (based on the residuals). An additional parameter
            - alpha can be specified to apply variance reduction, in order to account for spatial averaging. The parameter
            alpha refers to alpha = 1- Gamma^2. Typical values are:
                - alpha = 1     : full avaraging, uncertainty in the average
                - alpha = 0.75  : partly avaraging, regional data where 75% of the variance is assumed to average.
                - alpha = 0     : no averaging, uncertainty in single point value.

            """

            Z = (
                (std_intercept**2)
                + (x**2 * std_slope**2)
                + 2 * rho * x * std_intercept * std_slope
                + (1 - alpha) * residuals / (N - 2)
            )

            from scipy.stats import t

            tz = t(N - 2).ppf(quantile) * np.sqrt(Z)
            y = (slope * x + intercept) + tz

            return y

        return fit_params, other_params, regression_function
