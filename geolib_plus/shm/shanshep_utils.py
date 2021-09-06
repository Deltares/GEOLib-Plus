from math import log
from pydantic import BaseModel
from typing import Union, Optional
import numpy as np
from scipy.optimize import curve_fit, minimize


class ShanshepUtils(BaseModel):
    """
    Class contains shansep utilities for parameter determination following the methodology as described in
    :cite: `meer_2019`.
    """

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def __S_and_m_not_inputted(x, A, B):
        """ """
        return A * x + B

    @staticmethod
    def __S_inputted(X, A):
        """ """
        x, S = X
        return A * x + S

    @staticmethod
    def __m_inputted(X, B):
        """ """
        x, m = X
        return m * x + B

    @staticmethod
    def get_shanshep_parameters(
        OCR: Union[float, np.array],
        su: Union[float, np.array],
        sigma_effective: Union[float, np.array],
        S: Optional[float] = None,
        m: Optional[float] = None,
    ):
        """
        Determines shansep parameters s and m according to :cite: `meer_2019`.
        This is done by using simple linear regression.
        Parameter S can also be given as an input.
        """
        if (m is None) and (S is None):
            log_OCR = np.log(OCR)
            log_su_sigma = np.log(np.divide(su, sigma_effective))

            popt, _ = curve_fit(
                ShanshepUtils.__S_and_m_not_inputted, log_OCR, log_su_sigma, method="lm"
            )
            # summarize the parameter values
            m, log_S = popt
            S = 10 ** log_S
        elif (m is None) and (S is not None):
            log_OCR = np.log(OCR)
            log_su_sigma = np.log(np.divide(su, sigma_effective))

            popt, _ = curve_fit(
                ShanshepUtils.__S_inputted,
                (log_OCR, np.full(log_OCR.shape, np.log(S))),
                log_su_sigma,
                method="lm",
            )
            # summarize the parameter values
            m = popt[0]
        elif (m is not None) and (S is None):
            log_OCR = np.log(OCR)
            log_su_sigma = np.log(np.divide(su, sigma_effective))

            popt, _ = curve_fit(
                ShanshepUtils.__m_inputted,
                (log_OCR, np.full(log_OCR.shape, np.log(m))),
                log_su_sigma,
                method="lm",
            )
            # summarize the parameter values
            S = popt[0]

        return (S, m)
