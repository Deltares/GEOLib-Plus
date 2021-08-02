
# import packages

from typing import Iterable
from pydantic import BaseModel

import numpy as np
from scipy.stats import t
from scipy.stats import norm


class ProbUtils(BaseModel):

    @staticmethod
    def calculate_student_t_factor(ndof: int, quantile: float) -> float:
        """
        Gets student t factor from student t distribution

        :param ndof: number of degrees of freedom
        :param quantile: quantile where the student t factor should be calculated
        :return: Student t factor
        """

        return t(ndof).ppf(quantile)

    @staticmethod
    def correct_std_with_student_t(ndof: int, quantile: float, std: float, a: float) -> float:
        """
        Calculates corrected standard deviation at a quantile with the student-t factor

        :param ndof: number of degrees of freedom
        :param quantile: quantile where the student t factor should be calculated
        :param std: standard deviation
        :param a: spread factor, 0.75 for regional sample collection; 1.0 for local sample collection
        :return: corrected standard deviation
        """

        t_factor = ProbUtils.calculate_student_t_factor(ndof-1, quantile)
        norm_factor = norm.ppf(quantile)
        corrected_std = t_factor/norm_factor * std * np.sqrt((1-a) + (1/ndof))

        return corrected_std

    @staticmethod
    def get_mean_std_from_lognormal(log_mean: float, log_std: float, shift: float=0):
        """
        Calculates normal mean and standard deviation from the mean and std of LN(X)

        :param log_mean: mean of LN(X)
        :param log_std: std of LN(X)
        :param shift: shift from 0 of the log normal distribution
        :return: mean and std of X
        """

        mean = np.exp(log_mean + (log_std**2)/2) + shift
        std = np.sqrt((mean - shift)**2 * np.exp(log_std**2) - 1)

        return mean, std

    @staticmethod
    def get_log_mean_std_from_normal(mean, std, shift=0):
        """
        Calculates mean and standard deviation of LN(X) from the mean and std of X

        :param mean: mean of X
        :param std: std X
        :param shift: shift from 0 of the log normal distribution
        :return: mean and std of LN(X)
        """

        log_mean = np.log((mean - shift)**2 / (np.sqrt(std ** 2 + (mean-shift) ** 2)))
        log_std = np.sqrt(np.log((std/(mean-shift))**2 + 1))

        return log_mean, log_std

    @staticmethod
    def calculate_log_stats(data: Iterable):
        """
        Calculates mean and std of LN(X)

        :param data: dataset, X
        :return: mean and std of LN(X)
        """
        log_mean = np.sum(np.log(data))/ len(data)
        log_std = np.sqrt(np.sum(np.log(data)-log_mean)**2 / (len(data) - 1))

        return log_mean, log_std

    @staticmethod
    def calculate_normal_stats(data: Iterable):
        """
        Calculates mean and std of X

        :param data: dataset, X
        :return: mean and std of X
        """
        mean = np.sum(data)/ len(data)
        std = np.sqrt(np.sum(data-mean)**2 / (len(data) - 1))

        return mean, std

    @staticmethod
    def calculate_characteristic_value_from_dataset(data: Iterable, is_local: bool, is_low: bool,
                                                    is_log_normal: bool = True, char_quantile: float = 0.05):
        """
        Calculates the characteristic value of the dataset. A normal distribution or a lognormal distribution can be
        assumed for the dataset. The student-t distribution is taken into account.

        :param data: dataset, X
        :param is_local: true if data collection is local, false if data collection is regional
        :param is_low: true if low characteristic value is to be calculated, false if high characteristic value desired
        :param is_log_normal: True if a log normal distribution is assumed, false for normal distribution
        :param char_quantile: Quantile which is considered for the characteristic value

        :return: characteristic value of the dataset
        """

        direction_factor = -1 if is_low else 1

        # set spread reduction factor, 0.75 if data collection is regional, 1.0 if data collection is local
        if is_local:
            a = 1
        else:
            a = 0.75

        if is_log_normal:
            # calculate characteristic value from log normal distribution
            log_mean, log_std = ProbUtils.calculate_log_stats(data)
            log_std = ProbUtils.correct_std_with_student_t(len(data), char_quantile, log_std, a)

            x_kar = np.exp(log_mean + direction_factor*log_std)
        else:
            # calculate characteristic value from normal distribution
            mean, std = ProbUtils.calculate_normal_stats(data)
            std = ProbUtils.correct_std_with_student_t(len(data), char_quantile, std, a)

            x_kar = mean + direction_factor * std

        return x_kar


