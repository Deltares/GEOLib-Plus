from abc import abstractmethod
from pathlib import Path
from typing import Optional, Iterable, List, Type, Union
from pydantic import BaseModel
from copy import deepcopy
import numpy as np
import math

from scipy.optimize import minimize as sc_minimize

class DetermineNkt:
    nkt_mean: Optional[Union[np.ndarray, float]]
    nkt_std: Optional[Union[np.ndarray, float]]
    nkt_vc: Optional[Union[np.ndarray, float]]

    __nkt_std: Optional[Union[np.ndarray, float]]
    __nkt_vc: Optional[Union[np.ndarray, float]]

    class Config:
        arbitrary_types_allowed = True

    @property
    def nkt_std(self):
        return self.__nkt_std

    @nkt_std.setter
    def nkt_std(self, nkt_std):
        self.__nkt_vc = nkt_std / self.nkt_mean
        self.__nkt_std = nkt_std

    @property
    def nkt_vc(self):
        return self.__nkt_vc

    @nkt_vc.setter
    def nkt_vc(self, nkt_vc):
        self.__nkt_std = nkt_vc * self.nkt_mean
        self.__nkt_vc = nkt_vc

    def get_default_nkt(self, is_saturated: Optional[Union[np.ndarray, bool]]):

        saturated_nkt_mean = 20
        saturated_nkt_vc = 0.25
        unsaturated_nkt_mean = 60
        unsaturated_nkt_vc = 0.25

        if isinstance(is_saturated, bool):
            if is_saturated:
                self.nkt_mean = saturated_nkt_mean
                self.nkt_vc = saturated_nkt_vc
            else:
                self.nkt_mean = unsaturated_nkt_mean
                self.nkt_vc = unsaturated_nkt_vc

        else:
            self.nkt_mean = np.zeros(len(is_saturated))
            self.nkt_vc = np.zeros(len(is_saturated))

            self.nkt_mean[is_saturated] = saturated_nkt_mean
            self.nkt_vc[is_saturated] = saturated_nkt_vc

            self.nkt_mean[~is_saturated] = unsaturated_nkt_mean
            self.nkt_vc[~is_saturated] = unsaturated_nkt_vc


    def get_nkt_from_minimising_vc(self, su, q_net):

        minimisation_function = lambda mu_nkt:  np.sqrt(np.sum(((np.array(su) * mu_nkt)/np.array(q_net) -1)**2) / (len(su)-1))
        res = sc_minimize(minimisation_function, 0)

        self.nkt_mean = res.x
        self.nkt_vc = res.fun


    def get_nkt_from_statistics(self, su, q_net):

        nkt = np.array(q_net) / np.array(su)




determine_nkt = DetermineNkt()

determine_nkt.nkt_mean = 6
determine_nkt.nkt_std = 1

determine_nkt.nkt_vc = 0.5

determine_nkt.nkt_std =1

a=1+1

q_net = [300,300, 400, 700, 1200, 1400]
su = [20,21,24, 55, 64,60]

# import matplotlib.pyplot as plt
#
# plt.plot(q_net,su, 'o')
# plt.show()

determine_nkt.get_nkt_from_statistics(su, q_net)
# determine_nkt.get_nkt_from_minimising_vc(su, q_net)