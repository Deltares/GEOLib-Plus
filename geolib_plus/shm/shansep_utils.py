from pydantic import BaseModel
from typing import Union, Optional
import numpy as np
from scipy.optimize import curve_fit

from geolib_plus.shm.prob_utils import ProbUtils


class ShansepUtils(BaseModel):
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
        x, log_S = X
        return A * x + log_S

    @staticmethod
    def __m_inputted(X, B):
        """ """
        x, m = X
        return m * x + B

    @staticmethod
    def get_shansep_prob_parameters_with_linear_regression(
        OCR: Union[float, np.array],
        su: Union[float, np.array],
        sigma_effective: Union[float, np.array],
        S: Optional[float] = None,
        m: Optional[float] = None,
    ) -> ((float, float), (float, float)):
        """
        Determines shansep parameters s and m with linear regression according to :cite: `meer_2019`.
        Parameter S or m can also be given as an input.

        :param OCR: Union[float, np.array],
        :param su: Undrained shear strength
        :param sigma_effective: Effective stress
        :param S: S value
        :param m: m value
        :return: (mean and std of S), (mean and std of m)
        """
        if (m is None) and (S is None):
            log_OCR = np.log(OCR)
            log_su_sigma = np.log(np.divide(su, sigma_effective))

            popt, cov = curve_fit(
                ShansepUtils.__S_and_m_not_inputted, log_OCR, log_su_sigma, method="lm"
            )

            # summarize the parameter values
            m, log_S = popt
            std_m, std_log_s = np.sqrt(cov[0,0]), np.sqrt(cov[1, 1])

            # correct std with student t distribution
            std_m = ProbUtils.correct_std_with_student_t(len(log_su_sigma)-1, 0.05,std_m, 0.75)
            std_log_s = ProbUtils.correct_std_with_student_t(len(log_su_sigma) - 1, 0.05, std_log_s, 0.75)

            S, std_s = ProbUtils.get_mean_std_from_lognormal(log_S, std_log_s)

        elif (m is None) and (S is not None):
            log_OCR = np.log(OCR)
            log_su_sigma = np.log(np.divide(su, sigma_effective))

            popt, cov = curve_fit(
                ShansepUtils.__S_inputted,
                (log_OCR, np.full(log_OCR.shape, np.log(S))),
                log_su_sigma,
                method="lm",
            )
            # summarize the parameter values
            m = popt[0]
            std_m = np.sqrt(cov[0,0])

            # correct std with student t distribution
            std_m = ProbUtils.correct_std_with_student_t(len(log_su_sigma) - 1, 0.05, std_m, 0.75)
            std_s = 0

        elif (m is not None) and (S is None):
            log_OCR = np.log(OCR)
            log_su_sigma = np.log(np.divide(su, sigma_effective))

            popt, cov = curve_fit(
                ShansepUtils.__m_inputted,
                (log_OCR, np.full(log_OCR.shape, m)),
                log_su_sigma,
                method="lm",
            )
            # summarize the parameter values
            log_S = popt[0]
            std_log_s = np.sqrt(cov[0, 0])

            # correct std with student t distribution
            std_log_s = ProbUtils.correct_std_with_student_t(len(log_su_sigma) - 1, 0.05, std_log_s, 0.75)
            S, std_s = ProbUtils.get_mean_std_from_lognormal(log_S, std_log_s)
            std_m = 0

        else:
            # both s and m are given
            std_s = 0
            std_m = 0

        return (S, std_s), (m, std_m)
