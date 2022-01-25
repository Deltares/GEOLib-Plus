from enum import IntEnum
from typing import Iterable, Optional, Union

import numpy as np
from pydantic import BaseModel
from scipy.optimize import minimize as sc_minimize

from geolib_plus.shm.prob_utils import ProbUtils


class NktMethod(IntEnum):
    REGRESSION = 1
    STATISTICS = 2


class NktUtils(BaseModel):
    """
    This class contains utility functions which are used to determine Nkt and the characteristic value and probabilistic
    parameters. All methods are according to  :cite:`meer_2019`.
    """

    @staticmethod
    def get_default_nkt(
        is_saturated: Optional[Union[np.ndarray, bool]]
    ) -> (Union[np.ndarray, bool], Union[np.ndarray, bool]):
        r"""
        Gets default Nkt values.

        For saturated soil: mean Nkt is 20 and variation coefficient is 0.25
        For unsaturated soil: mean Nkt is 60 and variation coefficient is 0.25


        :param is_saturated: boolean or numpy array with booleans which indicate of the soil/soils is/are saturated

        :return: mean of Nkt, std of Nkt
        """

        # set mean and variation coefficient values of nkt
        saturated_nkt_mean = 20
        saturated_nkt_vc = 0.25
        unsaturated_nkt_mean = 60
        unsaturated_nkt_vc = 0.25

        # set singular Nkt mean and variation coefficient
        if isinstance(is_saturated, bool):
            if is_saturated:
                nkt_mean = saturated_nkt_mean
                nkt_vc = saturated_nkt_vc
            else:
                nkt_mean = unsaturated_nkt_mean
                nkt_vc = unsaturated_nkt_vc

        # set np array of Nkt means and variation coefficients
        else:
            nkt_mean = np.zeros(len(is_saturated))
            nkt_vc = np.zeros(len(is_saturated))

            nkt_mean[is_saturated] = saturated_nkt_mean
            nkt_vc[is_saturated] = saturated_nkt_vc

            nkt_mean[~is_saturated] = unsaturated_nkt_mean
            nkt_vc[~is_saturated] = unsaturated_nkt_vc

        # calculate Nkt standard deviation
        nkt_std = nkt_mean * nkt_vc

        return nkt_mean, nkt_std

    @staticmethod
    def get_nkt_stats_from_weighted_regression(
        su: np.ndarray, q_net: np.ndarray
    ) -> (float, float):
        r"""
        Gets Nkt statistics from weighted regression. With this method, the mean of Nkt and the variation coefficient
        of q_net/Nkt are found through weighted regression where the variation coefficient is minimised.

        The function to be minimised is:

        .. math::

            V_{\frac{q_{net}}{N_{kt}}.tot} = \sqrt{\frac{\sum{(\frac{s_{u,i} \cdot \mu_{N_{kt}}}{q_{net,i}} -1)}^{2}}{n-1}}


        :param su: iterable of undrained shear strength [kPa]
        :param q_net: iterable of net cone resistance [kPa]

        :return: mean of Nkt, variation coefficient of q_net/Nkt
        """

        # set minimisation function
        minimisation_function = lambda mu_nkt: np.sqrt(
            np.sum(((np.array(su) * mu_nkt) / np.array(q_net) - 1) ** 2) / (len(su) - 1)
        )

        # perform minimisation
        res = sc_minimize(minimisation_function, 0)

        # get Nkt mean and variation coefficient of  q_net/Nkt
        nkt_mean = res.x
        vc_qnet_nkt_tot = res.fun

        return nkt_mean, vc_qnet_nkt_tot

    @staticmethod
    def get_characteristic_value_nkt_from_weighted_regression(
        su: np.ndarray, q_net: np.ndarray, vc_loc: float = None
    ) -> float:
        r"""
        Gets characteristic value of nkt from weighted regression.

        The characteristic value of N_kt is calculated by using the following formula:

        .. math::

            N_{kt,kar} = \frac{\mu_{N_{kt}}}{(1-T^{0.05}_{n-1} \cdot V_{\frac{q_{net}}{N_{kt}},gem} \cdot \sqrt{1 + \frac{1}{n}})}


        :param su: iterable of undrained shear strength
        :param q_net: iterable of net cone resistance
        :param vc_loc: local variation coefficient of q_net/N_kt, if None: vc_loc = 0.5 vc_total

        :return: characteristic value of N_kt
        """

        # get mean Nkt and variation coefficient of q_net/nkt_total through weighted regression
        nkt_mean, vc_qnet_nkt_tot = NktUtils.get_nkt_stats_from_weighted_regression(
            su, q_net
        )

        # set vc_loc as 0.5 of variation coefficient of q_net/nkt_total if none is given
        if vc_loc is None:
            vc_loc = 0.5 * vc_qnet_nkt_tot

        # calculate average variation coefficient of q_net/nkt
        vc_average = np.sqrt(vc_qnet_nkt_tot ** 2 - vc_loc ** 2)

        # get number of tests
        n = len(su)

        # calculate student t factor at 95 percentile
        student_t_factor = ProbUtils.calculate_student_t_factor(n - 1, 0.95)

        # calculate characteristic value of Nkt
        nkt_char = nkt_mean / (1 - student_t_factor * vc_average * np.sqrt(1 + (1 / n)))

        return nkt_char

    @staticmethod
    def get_nkt_from_statistics(su: np.ndarray, q_net: np.ndarray) -> (float, float):
        r"""
        Calculates log mean of N_kt and total log standard deviation of N_kt through statistics

        :param su: iterable of undrained shear strength
        :param q_net: iterable of net cone resistance

        :return: mean of Ln Nkt, total standard deviation of Ln Nkt
        """

        # calculate Nkt
        nkt = np.array(q_net) / np.array(su)

        # get log mean and std
        log_nkt_mean, log_nkt_std_tot = ProbUtils.calculate_log_stats(nkt)

        return log_nkt_mean, log_nkt_std_tot

    @staticmethod
    def get_characteristic_value_nkt_from_statistics(
        su: np.ndarray, q_net: np.ndarray, std_loc: float = None
    ) -> float:
        r"""
        Gets characteristic value of N_kt through statistics.

        The characteristic value of N_kt is calculated by using the following formula:

        .. math::

            N_{kt,kar} = exp(\mu_{N_{kt}} + T^{0.05}_{n-1} \cdot \sigma_{Ln(N_{kt,gem}) \cdot \sqrt{1 + \frac{1}{n}}})


        :param su: iterable of undrained shear strength
        :param q_net: iterable of net cone resistance
        :param std_loc: local standard deviation of log N_kt, if None: std_loc = 0.5 std_total

        :return: characteristic value of N_kt
        """

        # get log nkt mean and total standard deviation from statistics
        log_nkt_mean, log_nkt_std_tot = NktUtils.get_nkt_from_statistics(su, q_net)

        # set local standard deviation
        if std_loc is None:
            std_loc = 0.5 * log_nkt_std_tot

        # calculate average standard deviation
        std_average = np.sqrt(log_nkt_std_tot ** 2 - std_loc ** 2)

        # get number of data points
        n = len(su)

        # get student t factor at 95 percentile
        student_t_factor = ProbUtils.calculate_student_t_factor(n - 1, 0.95)

        # calculate characteristic value of N_kt
        nkt_char = np.exp(
            log_nkt_mean + student_t_factor * std_average * np.sqrt(1 + 1 / n)
        )

        return nkt_char

    @staticmethod
    def get_prob_nkt_parameters_from_weighted_regression(
        su: np.ndarray, q_net: np.ndarray, vc_loc: float = None
    ) -> (float, float):
        r"""
        Get Nkt parameters for probabilistic analysis through weighted regression.

        Firstly, weighted regression is used to get the mean of the Nkt values and the variation coefficient of
        q_net / N_kt. Afterwards, the variation coefficient is adjusted for the number of samples with the student T
        distribution

        :param su: iterable of undrained shear strength
        :param q_net: iterable of net cone resistance
        :param vc_loc: local variation coefficient of q_net/N_kt, if None: vc_loc = 0.5 vc_total

        :return: mean of N_kt and variation coefficient of q_net/N_kt for probabilistic analysis
        """

        # calculate mean of the Nkt values and the variation coefficient of q_net / N_kt
        nkt_mean, vc_qnet_nkt_tot = NktUtils.get_nkt_stats_from_weighted_regression(
            su, q_net
        )

        # set local variation coefficient in case it is not given
        if vc_loc is None:
            vc_loc = 0.5 * vc_qnet_nkt_tot

        # calculate average variation coefficient
        vc_average = np.sqrt(vc_qnet_nkt_tot ** 2 - vc_loc ** 2)

        # calculate variation coefficient for probabilistic analysis
        vc_prob = ProbUtils.correct_std_with_student_t(len(su), 0.05, vc_average, 0)

        return nkt_mean, vc_prob

    @staticmethod
    def get_prob_nkt_parameters_from_statistics(
        su: np.ndarray, q_net: np.ndarray, log_std_loc: float = None
    ) -> (float, float):
        r"""
        Get Nkt parameters for probabilistic analysis through statistics.

        Firstly, statistics are used to get the mean and standard deviation of the log Nkt values.
        Afterwards, the mean and standard deviation for probabilistic analysis are calculated, where the student T
        distribution is taken into account.

        :param su: iterable of undrained shear strength
        :param q_net: iterable of net cone resistance
        :param log_std_loc: local standard deviation of log N_kt, if None: std_loc = 0.5 std_total

        :return: mean  of N_kt and variation coefficient of q_net/N_kt for probabilistic analysis
        """

        # get mean and std of log Nkt
        log_nkt_mean, log_nkt_std_tot = NktUtils.get_nkt_from_statistics(su, q_net)

        # set local standard deviation in case it is not given
        if log_std_loc is None:
            log_std_loc = 0.5 * log_nkt_std_tot

        # calculate average local standard deviation
        log_std_average = np.sqrt(log_nkt_std_tot ** 2 - log_std_loc ** 2)

        # adjust std of Log nkt with the student T distribution
        log_std_prob = ProbUtils.correct_std_with_student_t(
            len(su), 0.05, log_std_average, 0
        )

        # calculate mean and standard deviation of Nkt
        mean_nkt, std_nkt = ProbUtils.get_mean_std_from_lognormal(
            log_nkt_mean, log_std_prob
        )

        # calculate variation coefficient
        vc_qnet_nkt = std_nkt / mean_nkt

        return mean_nkt, vc_qnet_nkt
