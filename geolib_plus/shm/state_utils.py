import numpy as np
from pydantic import BaseModel


class StateUtils(BaseModel):
    @staticmethod
    def calculate_yield_stress(
        su: float, effective_stress: float, S: float, m: float
    ) -> float:
        """
        Calculates yield stress with the Shansep relation.

        .. math::

            \sigma_{vy}' = \sigma_{vi}' \cdot \frac{su}{\sigma_{vi}' \cdot S}^{\frac{1}{m}}

        :param su: Undrained Shear Strength [kPa] at the depth of interest
        :param effective_stress: Vertical In situ Effective Stress [kPa] at the depth of interest
        :param S: Normally Consolidated Undrained Shear Strength Ratio [-] (input from laboratory)
        :param m: Strength Increase Exponent [-] (input from laboratory)
        """

        return effective_stress * (su / (effective_stress * S)) ** (1 / m)

    @staticmethod
    def calculate_yield_stress_prob_parameters_from_cpt(
        effective_stress: float,
        q_net: float,
        mu_S: float,
        mu_m: float,
        mean_nkt: float,
        vc_q_net_nkt: float,
    ) -> (float, float):
        """
        Calculates probabilistic parameters of the yield stress, using Nkt and q_net data. Through the shansep relation
        the qnet_Nkt is used to calculate the yield stress.  The standard deviation of the yield stress is calculated
        by:

        .. math::

            \sigma_{vy}' = \frac{\frac{q_{net}}/{N_{kt}}}{\mu_{S}} * Vc_{ \frac{q_{net}}/{N_{kt}}}

        :param effective_stress: Vertical In situ Effective Stress [kPa] at the depth of interest
        :param q_net: Cone resistance corrected with water pressure [kPa] at the depth of interest
        :param mu_S: Mean normally Consolidated Undrained Shear Strength Ratio [-] (input from laboratory)
        :param mu_m: Mean Strength Increase Exponent [-] (input from laboratory)
        :param mean_nkt: Mean of nkt value
        :param vc_q_net_nkt: Variation coefficient of qnet/N_kt

        :return: mean of yield stress, std of yield stress
        """

        # calculates mean of qnet/Nkt
        mean_qnet_nkt = q_net / mean_nkt

        # calculate yield stress through shansep relation
        mu_yield = StateUtils.calculate_yield_stress(
            mean_qnet_nkt, effective_stress, mu_S, mu_m
        )

        # calculate standard deviation of yield stress
        sigma_yield = mean_qnet_nkt / mu_S * vc_q_net_nkt

        return mu_yield, sigma_yield

    @staticmethod
    def calculate_pop_prob_parameters_from_cpt(
        effective_stress: float,
        q_net: float,
        mu_S: float,
        mu_m: float,
        mean_nkt: float,
        vc_q_net_nkt: float,
    ) -> (float, float):
        """
        Calculates probabilistic parameters of the pre overburden pressure, using cpt data and lab data.
        Firstly the mean and standard deviation of the yield stress are calculated. The pop is then calculated by:

        .. math::

            \mu_{pop} = \mu_{\sigma_{vy}'} - \sigma'
            \sigma_{pop} = \sigma_{\sigma_{vy}'}

        :param effective_stress: Vertical In situ Effective Stress [kPa] at the depth of interest
        :param q_net: Cone resistance corrected with water pressure [kPa] at the depth of interest
        :param mu_S: Mean normally Consolidated Undrained Shear Strength Ratio [-] (input from laboratory)
        :param mu_m: Mean Strength Increase Exponent [-] (input from laboratory)
        :param mean_nkt: Mean of nkt value
        :param vc_q_net_nkt: Variation coefficient of qnet/N_kt

        :return: mean of pop, std of pop
        """

        (
            mu_yield,
            sigma_yield,
        ) = StateUtils.calculate_yield_stress_prob_parameters_from_cpt(
            effective_stress, q_net, mu_S, mu_m, mean_nkt, vc_q_net_nkt
        )

        mu_pop = mu_yield - effective_stress
        sigma_pop = sigma_yield

        return mu_pop, sigma_pop

    @staticmethod
    def calculate_ocr_prob_parameters_from_cpt(
        effective_stress: float,
        q_net: float,
        mu_S: float,
        mu_m: float,
        mean_nkt: float,
        vc_q_net_nkt: float,
    ) -> (float, float):
        """
        Calculates probabilistic parameters of the over consolidation ration, using cpt data and lab data.
        Firstly the mean and standard deviation of the yield stress are calculated. The ocr is then calculated by:

        .. math::

            \mu_{ocr} = \frac{\mu_{\sigma_{vy}'}}{\sigma'}
            \sigma_{ocr} = \frac{\sigma_{\sigma_{vy}'}}{\mu_{\sigma_{vy}'}} * \mu_{ocr}

        :param effective_stress: Vertical In situ Effective Stress [kPa] at the depth of interest
        :param q_net: Cone resistance corrected with water pressure [kPa] at the depth of interest
        :param mu_S: Mean normally Consolidated Undrained Shear Strength Ratio [-] (input from laboratory)
        :param mu_m: Mean Strength Increase Exponent [-] (input from laboratory)
        :param mean_nkt: Mean of nkt value
        :param vc_q_net_nkt: Variation coefficient of qnet/N_kt

        :return: mean of ocr, std of ocr
        """

        (
            mu_yield,
            sigma_yield,
        ) = StateUtils.calculate_yield_stress_prob_parameters_from_cpt(
            effective_stress, q_net, mu_S, mu_m, mean_nkt, vc_q_net_nkt
        )

        mu_ocr = mu_yield / np.mean(effective_stress)

        # Vc yield = Vc Ocr
        sigma_ocr = sigma_yield / mu_yield * mu_ocr

        return mu_ocr, sigma_ocr

    @staticmethod
    def calculate_characteristic_yield_stress(
        effective_stress: float,
        current_effective_stress: float,
        q_net: float,
        kar_S: float,
        kar_m: float,
        char_nkt,
    ) -> float:
        """
        Calculates characteristic value of yield stress, using cpt data and lab data.

        :param effective_stress: Vertical In situ Effective Stress [kPa] at the depth of interest during cpt
        :param current_effective_stress: Vertical current Effective Stress [kPa] at the depth of interest
        :param q_net: Cone resistance corrected with water pressure [kPa]
        :param kar_S: Characteristic value normally Consolidated Undrained Shear Strength Ratio [-] (input from laboratory)
        :param kar_m: Characteristic Strength Increase Exponent [-] (input from laboratory)
        :param char_nkt: characteristic value of nkt

        :return: characteristic value of yield stress
        """

        char_qnet_nkt = q_net / char_nkt

        yield_char = StateUtils.calculate_yield_stress(
            char_qnet_nkt, effective_stress, kar_S, kar_m
        )
        yield_char = max(yield_char, max(effective_stress, current_effective_stress))

        return yield_char

    @staticmethod
    def calculate_characteristic_pop(
        effective_stress: float,
        current_effective_stress: float,
        q_net: float,
        kar_S: float,
        kar_m: float,
        char_nkt,
    ) -> float:
        """
        Calculates characteristic value of pre overburden pressure, using cpt data and lab data.

        :param effective_stress: Vertical In situ Effective Stress [kPa] at the depth of interest during cpt
        :param current_effective_stress: Vertical current Effective Stress [kPa] at the depth of interest
        :param q_net: Cone resistance corrected with water pressure [kPa]
        :param kar_S: Characteristic value normally Consolidated Undrained Shear Strength Ratio [-] (input from laboratory)
        :param kar_m: Characteristic Strength Increase Exponent [-] (input from laboratory)
        :param char_nkt: characteristic value of nkt

        :return: characteristic value of pre overburden pressure
        """

        yield_char = StateUtils.calculate_characteristic_yield_stress(
            effective_stress, current_effective_stress, q_net, kar_S, kar_m, char_nkt
        )

        pop_char = max(yield_char - max(effective_stress, current_effective_stress), 0)
        return pop_char

    @staticmethod
    def calculate_characteristic_ocr(
        effective_stress: float,
        current_effective_stress: float,
        q_net: float,
        kar_S: float,
        kar_m: float,
        char_nkt,
    ) -> float:
        """
         Calculates characteristic value of over consolidation ratio, using cpt data and lab data.

         :param effective_stress: Vertical In situ Effective Stress [kPa] at the depth of interest during cpt
         :param current_effective_stress: Vertical current Effective Stress [kPa] at the depth of interest
         :param q_net: Cone resistance corrected with water pressure [kPa]
         :param kar_S: Characteristic value normally Consolidated Undrained Shear Strength Ratio [-] (input from laboratory)
         :param kar_m: Characteristic Strength Increase Exponent [-] (input from laboratory)
         :param char_nkt: characteristic value of nkt

        :return: characteristic value of over consolidation ratio
        """

        yield_char = StateUtils.calculate_characteristic_yield_stress(
            effective_stress, current_effective_stress, q_net, kar_S, kar_m, char_nkt
        )

        ocr_char = max(yield_char / max(effective_stress, current_effective_stress), 0)

        return ocr_char
