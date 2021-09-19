from typing import Optional, Iterable, Union
from pydantic import BaseModel
import numpy as np

from geolib_plus.shm.prob_utils import ProbUtils
from geolib_plus.shm.nkt_utils import NktMethod, NktUtils


class StateUtils(BaseModel):

    @staticmethod
    def calculate_yield_stress(su: float, effective_stress: float, S: float, m: float) -> float:
        """
        Calculates yield stress with the Shansep relation.

        .. math::

            \sigma_{vy}' = \sigma_{vi}' \cdot \frac{su}{\sigma_{vi}' \cdot S}^{\frac{1}{m}}

        :param su: Undrained Shear Strength [kPa] at the depth of interest
        :param effective_stress: Vertical In situ Effective Stress [kPa] at the depth of interest
        :param S: Normally Consolidated Undrained Shear Strength Ratio [-] (input from laboratory)
        :param m: Strength Increase Exponent [-] (input from laboratory)
        """
        return effective_stress * (su / (effective_stress * S)) ** 1 / m

    @staticmethod
    def calculate_yield_stress_prob_parameters_from_cpt(effective_stress: float, su: np.ndarray, q_net: np.ndarray,
                                                        mu_S: float, mu_m:float,
                                                        method: NktMethod = NktMethod.STATISTICS) -> (float, float):
        """
        Calculates probabilistic parameters of the yield stress, using cpt data.

        :param effective_stress: Vertical In situ Effective Stress [kPa] at the depth of interest
        :param su: Undrained Shear Strength [kPa] test set
        :param q_net: Cone resistance corrected with water pressure [kPa] test set
        :param mu_S: Mean normally Consolidated Undrained Shear Strength Ratio [-] (input from laboratory)
        :param mu_m: Mean Strength Increase Exponent [-] (input from laboratory)
        :param method: methodoly of calculation yield stress (NktMethod.STATISTICS or NktMethod.REGRESSION)
        """

        # calculates mean and variation coefficient of qnet/Nkt
        if method == NktMethod.STATISTICS:
            mean_qnet_nkt, vc_qnet_nkt = NktUtils.get_prob_nkt_parameters_from_statistics(su, q_net)
        elif method == NktMethod.REGRESSION:
            mean_qnet_nkt, vc_qnet_nkt = NktUtils.get_prob_nkt_parameters_from_weighted_regression(su, q_net)
        else:
            return None, None

        # calculate yield stress through shansep relation
        mu_yield = StateUtils.calculate_yield_stress(mean_qnet_nkt, effective_stress, mu_S, mu_m)

        # calculate standard deviation of yield stress
        sigma_yield = mean_qnet_nkt/mu_S * vc_qnet_nkt

        return mu_yield, sigma_yield

    @staticmethod
    def calculate_pop_prob_parameters_from_cpt(effective_stress: float, su: np.ndarray, q_net: np.ndarray, mu_S: float,
                                               mu_m: float, method=NktMethod.STATISTICS):

        mu_yield, sigma_yield = StateUtils.calculate_yield_stress_prob_parameters_from_cpt(effective_stress, su, q_net,
                                                                                           mu_S, mu_m, method=method)

        mu_pop = mu_yield - effective_stress
        sigma_pop = sigma_yield

        return mu_pop, sigma_pop

    @staticmethod
    def calculate_ocr_prob_parameters_from_cpt(effective_stress, su, q_net, mu_S, m,
                                               method=NktMethod.STATISTICS):

        mu_yield, sigma_yield = StateUtils.calculate_yield_stress_prob_parameters_from_cpt(effective_stress, su, q_net,
                                                                                           mu_S, m,
                                                                                           method=method)

        mu_ocr = mu_yield /np.mean(effective_stress)

        # Vc yield = Vc Ocr
        sigma_ocr = sigma_yield/mu_yield * mu_ocr

        return mu_ocr, sigma_ocr

    @staticmethod
    def calculate_characteristic_yield_stress(effective_stress, current_effective_stress, su, q_net, kar_S, kar_m,
                                              method=NktMethod.STATISTICS):

        if method == NktMethod.STATISTICS:
            mean_qnet_nkt, _ = NktUtils.get_prob_nkt_parameters_from_statistics(su, q_net)
        elif method == NktMethod.REGRESSION:
            mean_qnet_nkt, _ = NktUtils.get_prob_nkt_parameters_from_weighted_regression(su, q_net)
        else:
            return None, None

        yield_char = StateUtils.calculate_yield_stress(mean_qnet_nkt, effective_stress, kar_S, kar_m)
        yield_char = max(yield_char, max(effective_stress,current_effective_stress))

        return yield_char

    @staticmethod
    def calculate_characteristic_pop(effective_stress, current_effective_stress, su, q_net, kar_S, kar_m,
                                              method=NktMethod.STATISTICS):

        yield_char = StateUtils.calculate_characteristic_yield_stress(effective_stress, current_effective_stress, su, q_net, kar_S, kar_m,
                                              method=method)

        pop_char = max(yield_char - max(np.mean(effective_stress), np.mean(current_effective_stress)), 0)
        return pop_char

    @staticmethod
    def calculate_characteristic_ocr(effective_stress, current_effective_stress, su, q_net, kar_S, kar_m,
                                              method=NktMethod.STATISTICS):

        yield_char = StateUtils.calculate_characteristic_yield_stress(effective_stress, current_effective_stress, su, q_net, kar_S, kar_m,
                                              method=method)

        ocr_char = max(yield_char/ max(np.mean(effective_stress), np.mean(current_effective_stress)), 0)

        return ocr_char