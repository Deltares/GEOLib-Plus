.. tutorial_determine_nkt

Tutorial determine Nkt values using GEOLIB+
===========================================


The GEOLIB+ package allows for multiple methods to determine the Nkt value. All the methods are implemented as static
methods in the :class:`~geolib_plus.shm.nkt_utils.NktUtils` class and follow from  :cite:`meer_2019`. In this tutorial,
each method is explained.

Default Nkt
------------------

A default Nkt value can be retrieved with the function. :func:`~geolib_plus.shm.nkt_utils.NktUtils.get_default_nkt`. The
only input is the saturation of the soil. The saturation of the soil can be either a boolean or a numpy array of
booleans. Below an example is shown:

    .. code-block:: python

        from geolib_plus.shm.nkt_utils import NktUtils

        # get mean and standard deviation of Nkt
        is_soil_saturated = True
        default_mean_nkt, default_std_nkt = NktUtils.get_default_nkt(is_soil_saturated)

        # get array of mean and standard deviation of Nkt
        is_soil_saturated = np.array([True, True, False])
        default_nkt_mean, default_nkt_std = NktUtils.get_default_nkt(is_saturated)


Nkt from weighted regression
----------------------------

Another methodology to determine the Nkt, is weighted regression. Weighted regression can be used to get probabilistic
parameters of the Nkt value and also to retrieve the characteristic value. In both cases, a test set of Su values and
q_net values in the same soil type is required (q_net can be calculate in the
:class:`~geolib_plus.cpt_base_model.AbstractCPT` class).

With the function: :func:`~geolib_plus.shm.nkt_utils.NktUtils.get_prob_nkt_parameters_from_weighted_regression`, the mean Nkt
value and the variation coefficient of Nkt/q_net is determined, while taking into account the amount of tests in the
test set.

    .. code-block:: python

        from geolib_plus.shm.nkt_utils import NktUtils

        # get mean and standard deviation of Nkt
        su = np.array([42, 42, 28,29,34])  # in kPa
        q_net = np.array([260, 250, 210, 220, 260]) # in kPa

        mean_nkt, vc_qnet_nkt = NktUtils.get_prob_nkt_parameters_from_weighted_regression(su, q_net)

With the function :func:`~geolib_plus.shm.nkt_utils.NktUtils.get_characteristic_value_nkt_from_weighted_regression`, the
characteristic value of Nkt can be determined.

Nkt from statistics
----------------------------
Another method to retrieve the Nkt is to use statistics. With the function:
:func:`~geolib_plus.shm.nkt_utils.NktUtils.get_prob_nkt_parameters_from_statistics`, the mean Nkt and the variation
coefficient of qnet/Nkt can be determined. This method assumes a log-normal distribution of Nkt. And
it takes into account the amount of tests in the test set. Below an example is shown on how to use this function:

    .. code-block:: python

        from geolib_plus.shm.nkt_utils import NktUtils

        # get mean and standard deviation of Nkt
        su = np.array([42, 42, 28,29,34])  # in kPa
        q_net = np.array([260, 250, 210, 220, 260]) # in kPa

        mean_nkt, vc_qnet_nkt = NktUtils.get_prob_nkt_parameters_from_statistics(su, q_net)


With the function :func:`~geolib_plus.shm.nkt_utils.NktUtils.get_characteristic_value_nkt_from_statistics`, the
characteristic value of Nkt can be determined.
