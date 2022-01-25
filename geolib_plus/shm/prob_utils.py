# import packages

import numpy as np
from pydantic import BaseModel
from scipy.stats import norm, t


class ProbUtils(BaseModel):
    """
    Class contains probabilistic utilities for parameter determination following the methodology as described in
    :cite:`meer_2019`.
    """

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def calculate_student_t_factor(ndof: int, quantile: float) -> float:
        """
        Gets student t factor from student t distribution at a specific quantile.

        :param ndof: number of degrees of freedom
        :param quantile: quantile where the student t factor should be calculated
        :return: Student t factor
        """

        return t(ndof).ppf(quantile)

    @staticmethod
    def correct_std_with_student_t(
        ndof: int, quantile: float, std: float, a: float = 0
    ) -> float:
        r"""
        Calculates corrected standard deviation at a quantile with the student-t factor. This includes an optional
        spread reduction factor.
        The corrected standard deviation is calculated as follows:

        .. math::
            \sigma_{ln(x).prob} \approx  \frac{T^{0.05}_{n-1}}{u^{0.05}} \cdot \sigma_{ln(x)} \cdot \sqrt{(1-a) + \frac{1}{n}}


        where:

        :math:`T^{0.05}_{n-1}` is the student t factor at 5 :math:`\%` quantile with n-1 degrees of freedom

        :math:`u^{0.05}` is the value of the standard normal distribution at the  5 :math:`\%` quantile

        :math:`a` is the spread reduction factor; 0.75 if data collection is regional, 1.0 if data collection is local

        :param ndof: number of degrees of freedom
        :param quantile: quantile where the student t factor should be calculated
        :param std: standard deviation
        :param a: spread reduction factor, 0.75 for regional sample collection; 1.0 for local sample collection
        :return: corrected standard deviation

        """

        # get student t factor
        t_factor = ProbUtils.calculate_student_t_factor(ndof - 1, quantile)

        # get value at percentile for normal distribution
        norm_factor = norm.ppf(quantile)

        # calculate corrected standard deviation
        corrected_std = t_factor / norm_factor * std * np.sqrt((1 - a) + (1 / ndof))

        return corrected_std

    @staticmethod
    def calculate_prob_parameters_from_lognormal(
        data: np.ndarray, is_local: bool, quantile=0.05
    ):
        r"""
        Calculates probabilistic parameters mu and sigma from a lognormal dataset, as required in D-stability. This
        function takes into account spread reduction factor and the student t factor.

        Firstly the standard deviation of LN(X) is corrected for the number of test samples and type of samples (local
        or regional), secondly, the mean and standard deviation of X are calculated from the mean and corrected standard
        deviation of LN(X)

        :param data: dataset, X
        :param is_local: true if data collection is local, false if data collection is regional
        :param quantile: quantile where the student t factor should be calculated
        """
        log_mean, log_std = ProbUtils.calculate_log_stats(data)

        # set spread reduction factor, 0.75 if data collection is regional, 1.0 if data collection is local
        if is_local:
            a = 1
        else:
            a = 0.75

        corrected_std = ProbUtils.correct_std_with_student_t(
            len(data), quantile, log_std, a
        )

        mean_prob, std_prob = ProbUtils.get_mean_std_from_lognormal(
            log_mean, corrected_std
        )

        return mean_prob, std_prob

    @staticmethod
    def get_mean_std_from_lognormal(log_mean: float, log_std: float):
        """
        Calculates normal mean and standard deviation from the mean and std of LN(X)

        :param log_mean: mean of LN(X)
        :param log_std: std of LN(X)
        :return: mean and std of X
        """

        mean = np.exp(log_mean + (log_std ** 2) / 2)
        std = mean * np.sqrt(np.exp(log_std ** 2) - 1)

        return mean, std

    @staticmethod
    def get_log_mean_std_from_normal(mean: float, std: float):
        """
        Calculates mean and standard deviation of LN(X) from the mean and std of X

        :param mean: mean of X
        :param std: std X
        :return: mean and std of LN(X)
        """

        log_mean = np.log(mean ** 2 / (np.sqrt(std ** 2 + mean ** 2)))
        log_std = np.sqrt(np.log((std / mean) ** 2 + 1))

        return log_mean, log_std

    @staticmethod
    def calculate_log_stats(data: np.ndarray):
        """
        Calculates mean and std of LN(X)

        :param data: dataset, X
        :return: mean and std of LN(X)
        """
        log_mean = np.sum(np.log(data)) / len(data)
        log_std = np.sqrt(np.sum((np.log(data) - log_mean) ** 2) / (len(data) - 1))

        return log_mean, log_std

    @staticmethod
    def calculate_normal_stats(data: np.ndarray):
        """
        Calculates mean and std of X

        :param data: dataset, X
        :return: mean and std of X
        """
        mean = np.sum(data) / len(data)
        std = np.sqrt(np.sum((data - mean) ** 2) / (len(data) - 1))

        return mean, std

    @staticmethod
    def calculate_characteristic_value_from_prob_parameters(
        mean: float, std: float, n: int, char_quantile=0.05, a=0
    ):
        r"""
        Calculates characteristic values from probabilistic parameters.

        :param mean: mean of x
        :param std: standard deviation of x
        :param n: number of data points
        :param char_quantile: quantile of characteristic value (default = 0.05)
        :param a: spread reduction factor (default = 0)

        :return: characteristic value of X
        """

        # correct std for spread and amount of tests
        estimated_std = (
            ProbUtils.calculate_student_t_factor(n - 1, char_quantile)
            * std
            * np.sqrt((1 - a) + (1 / n))
        )

        x_kar = mean + estimated_std

        return x_kar

    @staticmethod
    def calculate_characteristic_value_from_dataset(
        data: np.ndarray,
        is_local: bool,
        is_low: bool,
        is_log_normal: bool = True,
        char_quantile: float = 0.05,
    ):
        r"""
        Calculates the characteristic value of the dataset. A normal distribution or a lognormal distribution can be
        assumed for the dataset. The student-t distribution is taken into account. And the spread reduction factor is
        taken into account.

        The characteristic value is calculated as follows:

         .. math::

            x_{kar} = exp(\mu_{ln(x)} \pm T^{0.05}_{n-1} \cdot \sigma_{ln(x)} \cdot \sqrt{(1-a) + \frac{1}{n}}

        where:

        :math:`x_{kar}` is the characteristic value

        :math:`T^{0.05}_{n-1}` is the student t factor at 5 :math:`\%` quantile with n-1 degrees of freedom

        :math:`a` is the spread reduction factor; 0.75 if data collection is regional, 1.0 if data collection is local

        :param data: dataset, X
        :param is_local: true if data collection is local, false if data collection is regional
        :param is_low: true if low characteristic value is to be calculated, false if high characteristic value desired
        :param is_log_normal: True if a log normal distribution is assumed, false for normal distribution
        :param char_quantile: Quantile which is considered for the characteristic value

        :return: characteristic value of the dataset
        """

        # direction_factor = -1 if is_low else 1

        if char_quantile > 0.5 and is_low:
            char_quantile = 1 - char_quantile
        elif char_quantile < 0.5 and not is_low:
            char_quantile = 1 - char_quantile

        # set spread reduction factor, 0.75 if data collection is regional, 1.0 if data collection is local
        if is_local:
            a = 1
        else:
            a = 0.75

        if is_log_normal:
            # calculate characteristic value from log normal distribution
            log_mean, log_std = ProbUtils.calculate_log_stats(data)

            # calculate log_x_kar
            log_x_kar = ProbUtils.calculate_characteristic_value_from_prob_parameters(
                log_mean, log_std, len(data), char_quantile, a
            )

            # calculate x_kar
            x_kar = np.exp(log_x_kar)
        else:
            # calculate characteristic value from normal distribution
            mean, std = ProbUtils.calculate_normal_stats(data)

            # calculate x_kar
            x_kar = ProbUtils.calculate_characteristic_value_from_prob_parameters(
                mean, std, len(data), char_quantile, a
            )

        return x_kar

    @staticmethod
    def calculate_std_from_vc(mean: float, vc: float):
        """
        Calculates standard deviation from variation coefficient

        :param mean: mean of distribution
        :param vc: variation coefficient of distribution

        """
        return mean * vc

    @staticmethod
    def calculate_vc_from_std(mean: float, std: float):
        """
        Calculates  variation coefficient from standard deviation

        :param mean: mean of distribution
        :param std: standard deviation of distribution

        """
        return std / mean
