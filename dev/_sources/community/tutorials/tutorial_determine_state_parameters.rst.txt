..tutorialcpt

Tutorial determine state parameters using GEOLIB+
===================================================

With the GEOLIB+ package it is possible to determine state parameters (OCR, POP and yield stress) of the soil, as described
in :cite:`meer_2019`. All the methods are implemented as static
methods in the :class:`~geolib_plus.shm.state_utils.StateUtils`.

Calculate yield stress, OverConsolidation Ratio and Pre Overburden Ratio
------------------------------------------------------------------------

With the function :func:`~geolib_plus.shm.state_utils.StateUtils.calculate_yield_stress_prob_parameters_from_cpt`
the mean and standard deviation of the yield stress can be determined. The input for this function is the effective
stress and the q_net at the depth of interest. Furthermore the function requires the mean S and mean m, which can be determined
as described in :any:`tutorial_determine_shansep_parameters`. Lastly, the function requires the mean Nkt and the
variation coefficient of q_net/ Nkt of the corresponding soil, which can be retrieved following
:any:`tutorial_determine_nkt`. Below an example is given on how to calculate the probabilistic parameters of the yield
stress.

    .. code-block:: python

        from geolib_plus.shm.nkt_utils import NktUtils

        effective_stress = 15.8
        q_net = 180
        mu_S = 0.38
        mu_m = 0.8
        mean_nkt = 16.01
        vc_qnet_nkt = 0.169

        mean_yield_stress, std_yield_stress = StateUtils.calculate_yield_stress_prob_parameters_from_cpt(effective_stress, q_net,
                                                                                                         mu_S, mu_m,
                                                                                                         mean_nkt, vc_qnet_nkt)

Calculating the probabilistic parameters of the POP and the OCR works in a similar wat, compared to calculating the yield stress.
In order to calculate the POP parameters, the function :func:`~geolib_plus.shm.state_utils.StateUtils.calculate_pop_prob_parameters_from_cpt`
can be used. In order to calculate the OCR parameters, the function: :func:`~geolib_plus.shm.state_utils.StateUtils.calculate_ocr_prob_parameters_from_cpt`.

It is also possible to calculate the characteristic values of the yield stress, POP and OCR. This can be done by using the following
corresponding functions: :func:`~geolib_plus.shm.state_utils.StateUtils.calculate_characteristic_yield_stress`,
:func:`~geolib_plus.shm.state_utils.StateUtils.calculate_characteristic_pop` and
:func:`~geolib_plus.shm.state_utils.StateUtils.calculate_characteristic_ocr`

