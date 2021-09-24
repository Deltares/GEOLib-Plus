..tutorialcpt

Tutorial determine Shansep parameters using GEOLIB+
===================================================

With the GEOLIB+ package it is possible to determine the shear strength ratio S and strength increase component m, using
linear regression as described in :cite:`meer_2019`. All the methods are implemented as static
methods in the :class:`~geolib_plus.shm.shansep_utils.ShansepUtils`. In this tutorial it is described how to retrieve
S and m from a lab data set.


Calculate S and m
------------------

With the function :func:`~geolib_plus.shm.shansep_utils.ShansepUtils.get_shansep_prob_parameters_with_linear_regression`
the mean and standard deviation of S and m can be determined, while taking into account the amount of tests in the
test data set. S is assumed to follow a log-normal distribution; m follows a normal distribution. The input for the
corresponding function is a set of OCR-values, Su values and effective stress values. The function also allows for a
given S or m. Below an example is given on how to calculate the S and m prob parameters.


    .. code-block:: python

        from geolib_plus.shm.shansep_utils import ShansepUtils

        ocr = np.array([1,1.78,1.59,2.89])
        su = np.array([21.3,19.9,71.8,57.9])
        sigma_eff = np.array([33,33,103,77])

        # calculate S and m mean and standard deviation and covariance matrix.
        (S_mean, s_std), (m_mean, m_std), covariance_matrix =
                    ShansepUtils.get_shansep_prob_parameters_with_linear_regression(ocr,su,sigma_eff)

        # calculate S and m mean and standard deviation and covariance matrix, while m is known.
        (S_mean_given_m, s_std_given_m), (m_mean_given_m, m_std_given_m), covariance_matrix =
                    ShansepUtils.get_shansep_prob_parameters_with_linear_regression(ocr,su,sigma_eff, m=0.8)

        # calculate S and m mean and standard deviation and covariance matrix, while m is known.
        (S_mean_given_S, s_std_given_S), (m_mean_give_S, m_std_given_S), covariance_matrix =
                    ShansepUtils.get_shansep_prob_parameters_with_linear_regression(ocr,su,sigma_eff, S=0.36)

With the function :func:`~geolib_plus.shm.shansep_utils.ShansepUtils.calculate_characteristic_shansep_parameters_with_linear_regression`,
the characteristic value of both S and m can be determined.

