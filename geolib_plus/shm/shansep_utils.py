from typing import Optional, Union

import numpy as np
from pydantic import BaseModel
from scipy.optimize import curve_fit

from geolib_plus.shm.prob_utils import ProbUtils


class ShansepUtils(BaseModel):
    """
    Class contains shansep utilities for parameter determination following the methodology as described in
    :cite:`meer_2019`.
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
    def calculate_characteristic_shansep_parameters_with_linear_regression(
        OCR: Union[float, np.array],
        su: Union[float, np.array],
        sigma_effective: Union[float, np.array],
        S: Optional[float] = None,
        m: Optional[float] = None,
    ) -> (float, float):
        """
        Calculates characteristic values of parameters s and m, according to :cite:`meer_2019`. This methodology
        assumes a log normal distribution of S and a normal distribution of m
        Optionally with a given S or m.

        :param OCR: Union[float, np.array], over consolidation ratio [-]
        :param su: Undrained shear strength [kPa]
        :param sigma_effective: Effective stress [kPa]
        :param S: S value, shear strength ratio [-]
        :param m: m value, strength increase component [-]
        :return: characteristic value of S and m

        """

        # get prob parameters of S and m
        (
            (S, std_s),
            (m, std_m),
            covariance_matrix,
        ) = ShansepUtils.get_shansep_prob_parameters_with_linear_regression(
            OCR, su, sigma_effective, S, m
        )

        # calculate mean log s and std log s
        log_s, log_std_s = ProbUtils.get_log_mean_std_from_normal(S, std_s)

        # calculate characteristic log s
        log_s_char = ProbUtils.calculate_characteristic_value_from_prob_parameters(
            log_s, log_std_s, len(OCR), char_quantile=0.05, a=0
        )

        # calculate characteristic S and m
        S_char = np.exp(log_s_char)
        m_char = ProbUtils.calculate_characteristic_value_from_prob_parameters(
            m, std_m, len(OCR), char_quantile=0.05, a=0
        )

        return S_char, m_char

    @staticmethod
    def get_shansep_prob_parameters_with_linear_regression(
        OCR: Union[float, np.array],
        su: Union[float, np.array],
        sigma_effective: Union[float, np.array],
        S: Optional[float] = None,
        m: Optional[float] = None,
    ) -> ((float, float), (float, float), np.ndarray):
        """
        Determines shansep parameters s and m with linear regression according to :cite:`meer_2019`. This methodology
        assumes a log normal distribution of S and a normal distribution of m. Parameter S or m can also be given as an
        input.

        :param OCR: Union[float, np.array], over consolidation ratio [-]
        :param su: Undrained shear strength [kPa]
        :param sigma_effective: Effective stress [kPa]
        :param S: S value, shear strength ratio [-]
        :param m: m value, strength increase component [-]
        :return: (mean and std of S), (mean and std of m), covariance matrix
        """

        # initialise covariance matrix
        covariance_matrix = np.zeros((2, 2))

        # calculate S and m
        if (m is None) and (S is None):
            log_OCR = np.log(OCR)
            log_su_sigma = np.log(np.divide(su, sigma_effective))

            popt, cov = curve_fit(
                ShansepUtils.__S_and_m_not_inputted, log_OCR, log_su_sigma, method="lm"
            )

            # summarize the parameter values
            m, log_S = popt
            std_m, std_log_s = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])

            # correct std with student t distribution
            std_m = ProbUtils.correct_std_with_student_t(
                len(log_su_sigma) - 1, 0.05, std_m, 0.75
            )
            std_log_s = ProbUtils.correct_std_with_student_t(
                len(log_su_sigma) - 1, 0.05, std_log_s, 0.75
            )

            S, std_s = ProbUtils.get_mean_std_from_lognormal(log_S, std_log_s)

            covariance_matrix = cov

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
            std_m = np.sqrt(cov[0, 0])

            # correct std with student t distribution
            std_m = ProbUtils.correct_std_with_student_t(
                len(log_su_sigma) - 1, 0.05, std_m, 0.75
            )
            std_s = 0

            covariance_matrix[0, 0] = cov[0, 0]

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
            std_log_s = ProbUtils.correct_std_with_student_t(
                len(log_su_sigma) - 1, 0.05, std_log_s, 0.75
            )
            S, std_s = ProbUtils.get_mean_std_from_lognormal(log_S, std_log_s)
            std_m = 0

            covariance_matrix[1, 1] = cov[0, 0]

        else:
            # both s and m are given
            std_s = 0
            std_m = 0

        return (S, std_s), (m, std_m), covariance_matrix
