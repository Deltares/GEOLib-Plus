import os
from enum import IntEnum
from pathlib import Path
from typing import Iterable, List, Union

import more_itertools as mit
import numpy as np
import shapefile
from pydantic import BaseModel
from shapely.geometry import Point, Polygon

from geolib_plus.cpt_base_model import AbstractCPT, AbstractInterpretationMethod
from geolib_plus.cpt_utils import (
    NEN_classification,
    NetCDF,
    ceil_value,
    merge_thickness,
    n_iter,
    resource_path,
)


class UnitWeightMethod(IntEnum):
    ROBERTSON = 1
    LENGKEEK = 2


class OCRMethod(IntEnum):
    ROBERTSON = 1
    MAYNE = 2


class ShearWaveVelocityMethod(IntEnum):
    ROBERTSON = 1
    MAYNE = 2
    ANDRUS = 3
    ZANG = 4
    AHMED = 5


class RelativeDensityMethod(IntEnum):
    BALDI = 1
    KULHAWY = 2
    KULHAWY_SIMPLE = 3


class RobertsonCptInterpretation(AbstractInterpretationMethod, BaseModel):
    r"""
    Robertson soil classification.

    Classification of soils according to Robertson chart.

    .. _element:
    .. figure:: ./../_static/robertson.png
        :width: 350px
        :align: center
        :figclass: align-center
    """

    unitweightmethod: UnitWeightMethod = UnitWeightMethod.ROBERTSON
    ocrmethod: OCRMethod = OCRMethod.ROBERTSON
    shearwavevelocitymethod: ShearWaveVelocityMethod = ShearWaveVelocityMethod.ROBERTSON
    data: AbstractCPT = None
    gamma: Iterable = []
    polygons: Iterable = []
    path_to_water_level_file: Union[str, Path] = Path(
        Path(__file__).parent, "resources"
    )
    name_water_level_file: str = "peilgebieden_jp_250m.nc"
    user_defined_water_level: bool = False

    def interpret(self, data: AbstractCPT):
        """
        Function that interprets the cpt inputs.
        Lithology for each layer is determined according to
        the robertson's method. Note that the pre_process method
        should be run before the interpret method.
        """
        # validate that interpretation can be run
        data.has_points_with_error()
        data.are_data_available_interpretation()
        data.has_duplicated_depth_values()
        data.check_if_lists_have_the_same_size()

        self.data = data
        MPa_to_kPa = 1000
        self.data.tip = self.data.tip * MPa_to_kPa
        self.data.friction = self.data.friction * MPa_to_kPa

        min_layer_thickness = 0.01
        # compute qc
        self.qt_calc()

        # compute unit weight
        # method = 'Robertson' (Default) or 'Lengkeek'
        # gamma_min = 10.5, gamma_max = 22 Defaults
        self.gamma_calc(method=self.unitweightmethod)

        # compute density
        self.rho_calc()

        # compute water pressure level # This requires a NHI file perhaps make it optional.
        self.pwp_level_calc()

        # compute stresses: total, effective and pore water pressures
        self.stress_calc()

        # compute Qtn and Fr
        self.norm_calc()

        # compute lithology
        self.lithology_calc()

        # compute IC
        self.IC_calc()

        # compute NEN values
        self.NEN_calc()

        # compute shear wave velocity and shear modulus
        # method == "Robertson" (Default|"Mayne"|"Andrus"|"Zang"|"Ahmed")
        self.vs_calc(method=self.shearwavevelocitymethod)

        # compute damping
        # method = "Mayne"|"Robertson" (Default)
        # d_min = 2
        # Cu = 2.
        # D50 = 0.2
        # Ip = 40
        # freq = 1.
        self.damp_calc(method=self.ocrmethod)

        # compute Poisson ratio
        self.poisson_calc()

        # compute Young modulus
        self.young_calc()

        # compute permeability
        self.permeability_calc()

        # compute clean sand equivalent normalised cone resistance
        self.norm_cone_resistance_clean_sand_calc()

        # compute state parameter
        self.state_parameter_calc()

        # filter values
        # lithologies = [""]
        # key = ""
        # value = 0
        self.filter()

        # merge the layers thickness
        (
            self.data.depth_merged,
            self.data.index_merged,
            self.data.lithology_merged,
        ) = merge_thickness(cpt_data=self.data, min_layer_thick=min_layer_thickness)

        self.data.tip = self.data.tip / MPa_to_kPa
        self.data.friction = self.data.friction / MPa_to_kPa
        return self.data

    def soil_types(
        self,
        path_shapefile: Path = Path("resources"),
        model_name: str = "Robertson",
    ):
        r"""
        Function that read shapes from shape file and passes them as Polygons.

        :param path_shapefile: Path to the shapefile
        :param model_name: Name of model and shapefile
        :return: list of the polygons defining the soil types
        """

        path_shapefile = resource_path(
            Path(Path(__file__).parent, path_shapefile, model_name)
        )

        # read shapefile
        sf = shapefile.Reader(str(path_shapefile))
        list_of_polygons = []
        for polygon in list(sf.iterShapes()):
            list_of_polygons.append(Polygon(polygon.points))
        self.polygons = list_of_polygons

    def lithology(self, Qtn: Iterable, Fr: Iterable):
        r"""
        Identifies lithology of CPT points, following Robertson and Cabal :cite:`robertson_cabal_2014`.

        Parameters
        ----------
        :return: lithology array, Qtn, Fr
        """

        lithology_array = [""] * len(Qtn)
        coords = np.zeros((len(Qtn), 2))

        # determine into which soil type the point is
        for i in range(len(Qtn)):
            pnt = Point(Fr[i], Qtn[i])
            aux = []
            for polygon in self.polygons:
                aux.append(polygon.contains(pnt))

            # check if point is within a boundary
            if all(not x for x in aux):
                aux = []
                for polygon in self.polygons:
                    aux.append(polygon.touches(pnt))

            idx = np.where(np.array(aux))[0][0]
            lithology_array[i] = str(idx + 1)
            coords[i] = [Fr[i], Qtn[i]]

        return lithology_array, np.array(coords)

    def lithology_calc(self):
        r"""
        Lithology calculation.

        :param soil_classification: shape file with soil classification
        """

        # call object
        self.soil_types()
        lithology, points = self.lithology(
            np.array(self.data.Qtn), np.array(self.data.Fr)
        )

        # assign to variables
        self.data.lithology = lithology
        self.data.litho_points = points

    def pwp_level_calc(self):
        r"""
        Computes the estimated pwp level for the cpt coordinate.
        If the user has not defined a pwp in the cpt class.

        """

        # If pwp is not None then the pore pressure should
        # be defined
        if not (self.user_defined_water_level):
            # define the path for the shape file
            path_pwp = Path(
                self.path_to_water_level_file,
                self.name_water_level_file,
            )

            # check if the file is nc file
            if not (self.name_water_level_file.split(".")[-1] == "nc"):
                raise TypeError("File should be NetCDF format : %s" % path_pwp)

            # check if file exists
            if not (path_pwp.is_file()):
                raise FileNotFoundError("File does not exist: %s" % path_pwp)

            # open file
            pwp = NetCDF()
            pwp.read_cdffile(path_pwp)
            pwp.query(self.data.coordinates[0], self.data.coordinates[1])
            self.data.pwp = pwp.NAP_water_level
        else:
            # check that there is a water level value defined
            if self.data.pwp is None:
                raise ValueError(
                    "Value of water level was not defined. Please input water level (pwp) or set user_defined_water_level to False"
                )

    def gamma_calc(
        self,
        method: UnitWeightMethod = UnitWeightMethod.ROBERTSON,
        gamma_min: float = 10.5,
        gamma_max: float = 22,
    ):
        r"""
        Computes unit weight.

        Computes the unit weight following Robertson and Cabal :cite:`robertson_cabal_2014`.
        If unit weight is infinity, it is set to gamma_limit.
        The formula for unit weight is:

        .. math::

            \gamma = (0.27 \log(R_{f}) + 0.36 \log\left(\frac{q_{t}}{Pa}\right) + 1.236) * \gamma_{w}

        Alternative method of Lengkeek et al. :cite:`lengkeek_2018`:

        .. math::

            \gamma = \gamma_{sat,ref} - \beta
            \left( \frac{\log \left( \frac{q_{t,ref}}{q_{t}} \right)}{\log \left(\frac{R_{f,ref}}{R_{f}}\right)} \right)

        Parameters
        ----------
        :param method: (optional) Method to compute unit weight. Default is Robertson
        :param gamma_max: (optional) Maximum gamma. Default is 22
        :param gamma_min: (optional) Minimum gamma. Default is 10.5
        """

        # ignore divisions warnings
        np.seterr(divide="ignore", invalid="ignore", over="print")

        # calculate unit weight according to Robertson & Cabal 2015
        if method == UnitWeightMethod.ROBERTSON:
            aux = (
                0.27 * np.log10(self.data.friction_nbr)
                + 0.36 * np.log10(np.array(self.data.qt) / self.data.Pa)
                + 1.236
            )
            # set lower limit
            aux = ceil_value(aux, gamma_min / self.data.g)
            # set higher limit
            aux[np.abs(aux) >= gamma_max] = gamma_max / self.data.g
            # assign gamma
            self.gamma = aux * self.data.g

        elif method == UnitWeightMethod.LENGKEEK:
            aux = 19.0 - 4.12 * np.log10(5000.0 / np.array(self.data.qt)) / np.log10(
                30.0 / self.data.friction_nbr
            )
            # if nan: aux is 19
            aux[np.isnan(aux)] = 19.0
            # set lower limit
            aux = ceil_value(aux, gamma_min)
            # set higher limit
            aux[np.abs(aux) >= gamma_max] = gamma_max
            # assign gamma
            self.gamma = aux

    def NEN_calc(self):
        r"""
        Computes the consistency according to the NEN 997-1:2016

        The NEN method is based on the tip resistance.
        """

        # Correction of the effective stress
        Cq0 = (self.data.Pa / self.data.effective_stress) ** 0.67
        # calculation of the corrected tip resistance
        qcdq = self.data.qt * Cq0

        # identify the class from NEN table
        classification = NEN_classification()
        classification.soil_types()
        NEN_result = classification.information(qcdq, self.data.lithology)

        # assign to variables
        self.data.litho_NEN = NEN_result["litho_NEN"]
        self.data.E_NEN = NEN_result["E_NEN"]
        self.data.cohesion_NEN = NEN_result["cohesion_NEN"]
        self.data.fr_angle_NEN = NEN_result["fr_angle_NEN"]

    def rho_calc(self):
        r"""
        Computes density of soil.

        The formula for density is:

        .. math::

            \rho = \frac{\gamma}{g}
        """

        self.data.rho = self.gamma * 1000.0 / self.data.g

    def stress_calc(self):
        r"""
        Computes total and effective stress
        """

        # compute depth diff
        z = np.diff(np.abs((self.data.depth - self.data.depth[0])))
        z = np.append(z, z[-1])
        # total stress
        self.data.total_stress = np.cumsum(self.gamma * z) + self.data.depth[
            0
        ] * np.mean(self.gamma[:10])
        # compute pwp
        # determine location of phreatic line: it cannot be above the CPT depth
        z_aux = np.min(
            [self.data.pwp, abs(self.data.depth_to_reference[0]) + self.data.depth[0]]
        )
        pwp = (z_aux - abs(self.data.depth_to_reference)) * self.data.g
        # no suction is allowed
        pwp[pwp <= 0] = 0
        # compute effective stress
        self.data.effective_stress = self.data.total_stress - pwp
        # if effective stress is negative -> effective stress = 0
        self.data.effective_stress[self.data.effective_stress <= 0] = 0

    def norm_calc(self, n_method: bool = False):
        r"""
        normalisation of qc and friction into Qtn and Fr, following Robertson and Cabal :cite:`robertson_cabal_2014`.

        .. math::

            Q_{tn} = \left(\frac{q_{t} - \sigma_{v0}}{Pa} \right) \left(\frac{Pa}{\sigma_{v0}'}\right)^{n}

            F_{r} = \frac{f_{s}}{q_{t}-\sigma_{v0}} \cdot 100


        Parameters
        ----------
        :param n_method: (optional) parameter *n* stress exponent. Default is n computed in an iterative way.
        """

        # normalisation of qc and friction into Qtn and Fr: following Robertson and Cabal (2014)

        # iteration around n to compute IC
        # start assuming n=1 for IC calculation
        n = np.ones(len(self.data.tip))

        # switch for the n calculation. default is iterative process
        if not n_method:
            tolerance = 1.0e-12
            error = 1
            max_iterations = 10000
            iteration = 0
            while error >= tolerance:
                # if did not converge
                if iteration >= max_iterations:
                    n = np.ones(len(self.data.tip)) * 0.5
                    break
                n1 = n_iter(
                    n,
                    self.data.tip,
                    self.data.friction_nbr,
                    self.data.effective_stress,
                    self.data.total_stress,
                    self.data.Pa,
                )
                error = np.linalg.norm(n1 - n) / np.linalg.norm(n1)
                n = n1
                iteration += 1
        else:
            n = np.ones(len(self.data.tip)) * 0.5

        # parameter Cn
        Cn = (self.data.Pa / self.data.effective_stress) ** n
        # calculation Q and F
        Q = (self.data.tip - self.data.total_stress) / self.data.Pa * Cn
        F = self.data.friction / (self.data.tip - self.data.total_stress) * 100
        # Q and F cannot be negative. if negative, log10 will be infinite.
        # These values are limited by the contours of soil behaviour of Robertson
        Q[Q <= 1.0] = 1.0
        F[F <= 0.1] = 0.1
        Q[Q >= 1000.0] = 1000.0
        F[F >= 10.0] = 10.0
        self.data.Qtn = Q
        self.data.Fr = F
        self.data.n = n

    def IC_calc(self):
        r"""
        IC, following Robertson and Cabal :cite:`robertson_cabal_2014`.

        .. math::

            I_{c} = \left[ \left(3.47 - \log\left(Q_{tn}\right) \right)^{2} + \left(\log\left(F_{r}\right) + 1.22 \right)^{2} \right]^{0.5}

        """

        # IC: following Robertson and Cabal (2015)
        # compute IC
        self.data.IC = (
            (3.47 - np.log10(self.data.Qtn)) ** 2.0
            + (np.log10(self.data.Fr) + 1.22) ** 2.0
        ) ** 0.5

    def vs_calc(
        self, method: ShearWaveVelocityMethod = ShearWaveVelocityMethod.ROBERTSON
    ):
        r"""
        Shear wave velocity and shear modulus. The following methods are available:

        * Robertson and Cabal :cite:`robertson_cabal_2014`:

        .. math::

            v_{s} = \left( \alpha_{vs} \cdot \frac{q_{t} - \sigma_{v0}}{Pa} \right)^{0.5}

            \alpha_{vs} = 10^{0.55 I_{c} + 1.68}

            G_{0} = \frac{\gamma}{g} \cdot v_{s}^{2}

        * Mayne :cite:`mayne_2007`:

        .. math::

            v_{s} = e^{\frac{\gamma_{sat} + 4.03}{4.17}} \cdot \left( \frac{\sigma_{v0}'}{\sigma_{atm}} \right)^{0.25}

            v_{s} = 118.8 \cdot \log \left(f_{s} \right) + 18.5

        * Andrus *et al.* :cite:`andrus_2007`:

        .. math::

            v_{s} = 2.27 \cdot q_{t}^{0.412} \cdot I_{c}^{0.989} \cdot D^{0.033} \cdot ASF  (Holocene)

            v_{s} = 2.62 \cdot q_{t}^{0.395} \cdot I_{c}^{0.912} \cdot D^{0.124} \cdot SF   (Pleistocene)

        * Zhang and Tong :cite:`zhang_2017`:

        .. math::

            v_{s} = 10.915 \cdot q_{t}^{0.317} \cdot I_{c}^{0.210} \cdot D^{0.057} \cdot SF^{a}  (Holocene)

        * Ahmed :cite:`ahmed_2017`:

        .. math::

            v_{s} = 1000 \cdot e^{-0.887 \cdot I_{c}} \cdot \left( \left(1 + 0.443 \cdot F_{r} \right) \cdot \left(\frac{\sigma'_{v}}{p_{a}} \right) \cdot \left(\frac{\gamma_{w}}{\gamma} \right) \right)^{0.5}
        """
        if method == ShearWaveVelocityMethod.ROBERTSON:
            # vs: following Robertson and Cabal (2015)
            alpha_vs = 10 ** (0.55 * self.data.IC + 1.68)
            vs = alpha_vs * (self.data.qt - self.data.total_stress) / self.data.Pa
            vs = ceil_value(vs, 0)
            self.data.vs = vs ** 0.5
            self.data.G0 = self.data.rho * self.data.vs ** 2
        elif method == ShearWaveVelocityMethod.MAYNE:
            # vs: following Mayne (2006)
            vs = 118.8 * np.log10(self.data.friction) + 18.5
            self.data.vs = ceil_value(vs, 0)
            self.data.G0 = self.data.rho * self.data.vs ** 2
        elif method == ShearWaveVelocityMethod.ANDRUS:
            # vs: following Andrus (2007)
            vs = (
                2.27
                * self.data.qt ** 0.412
                * self.data.IC ** 0.989
                * self.data.depth ** 0.033
                * 1
            )
            self.data.vs = ceil_value(vs, 0)

            self.data.G0 = self.data.rho * self.data.vs ** 2
        elif method == ShearWaveVelocityMethod.ZANG:
            # vs: following Zang & Tong (2017)
            vs = (
                10.915
                * self.data.qt ** 0.317
                * self.data.IC ** 0.210
                * self.data.depth ** 0.057
                * 0.92
            )
            self.data.vs = ceil_value(vs, 0)

            self.data.G0 = self.data.rho * self.data.vs ** 2
        elif method == ShearWaveVelocityMethod.AHMED:
            vs = (
                1000.0
                * np.exp(-0.887 * self.data.IC)
                * (
                    1.0
                    + 0.443
                    * self.data.Fr
                    * self.data.effective_stress
                    / self.data.Pa
                    * self.data.g
                    / self.gamma
                )
                ** 0.5
            )
            self.data.vs = ceil_value(vs, 0)
            self.data.G0 = self.data.rho * self.data.vs ** 2

    def young_calc(self):
        r"""
        Young modulus calculation.

        Computes the Young modulus:

        .. math::
            E = 2 \cdot G (1 + \mu)

        """

        self.data.E0 = 2 * self.data.G0 * (1 + self.data.poisson)

    def damp_calc(
        self,
        method: OCRMethod = OCRMethod.MAYNE,
        d_min: float = 2,
        Cu: float = 2.0,
        D50: float = 0.2,
        Ip: float = 40.0,
        freq: float = 1.0,
    ):
        r"""
        Damping calculation.

        For clays and peats, the damping is assumed as the minimum damping following Darendeli :cite:`darendeli_2001`.

        .. math::

            D_{min} = \left(0.8005 + 0.0129 \cdot PI \cdot OCR^{-0.1069} \right) \cdot \sigma_{v0}'^{-0.2889} \cdot \left[ 1 + 0.2919 \ln \left( freq \right) \right]

        The OCR can be computed according to Mayne :cite:`mayne_2007` or Robertson and Cabal :cite:`robertson_cabal_2014`.

        .. math::

            OCR_{Mayne} = 0.33 \cdot \frac{q_{t} - \sigma_{v0}}{\sigma_{v0}'}

            OCR_{Rob} = 0.25 \left(Q_{t}\right)^{1.25}

        For sand the damping is assumed as the minimum damping following Menq :cite`menq_2003`.

        .. math::
            D_{min} = 0.55 \cdot C_{u}^{0.1} \cdot d_{50}^{-0.3} \cdot  \left(\frac{\sigma'_{v}}{p_{a}} \right)^-0.08

        Parameters
        ----------
        :param method: (optional) Method for calculation of OCR. Default is Mayne
        :param d_min: (optional) Minimum damping. Default is 2%
        :param Cu: (optional) Coefficient of uniformity. Default is 2.0
        :param D50: (optional) Median grain size. Default is 0.2 mm
        :param Ip: (optional) Plasticity index. Default is 40
        :param freq: (optional) Frequency. Default is 1 Hz
        """
        # assign size to damping
        self.data.damping = np.zeros(len(self.data.lithology)) + d_min
        OCR = np.zeros(len(self.data.lithology))

        for i, lithology_index in enumerate(self.data.lithology):
            # if  clay
            if (
                lithology_index == "3"
                or lithology_index == "4"
                or lithology_index == "5"
            ):
                if method == OCRMethod.MAYNE:
                    OCR[i] = (
                        0.33
                        * (self.data.qt[i] - self.data.total_stress[i])
                        / self.data.effective_stress[i]
                    )
                elif method == OCRMethod.ROBERTSON:
                    OCR[i] = 0.25 * self.data.Qtn[i] ** 1.25

                self.data.damping[i] = (
                    (0.8005 + 0.0129 * Ip * OCR[i] ** (-0.1069))
                    * (self.data.effective_stress[i] / self.data.Pa) ** (-0.2889)
                    * (1 + 0.2919 * np.log(freq))
                )

            # if peat
            elif lithology_index == "1" or lithology_index == "2":
                # same as clay: OCR=1 IP=100
                self.data.damping[i] = (
                    2.512 * (self.data.effective_stress[i] / self.data.Pa) ** -0.2889
                )
            # if sand
            else:
                self.data.damping[i] = (
                    0.55
                    * Cu ** 0.1
                    * D50 ** -0.3
                    * (self.data.effective_stress[i] / self.data.Pa) ** -0.08
                )

        # limit the damping (when stress is zero damping is infinite)
        self.data.damping[self.data.damping == np.inf] = 100
        # damping units -> dimensionless
        self.data.damping /= 100

    def poisson_calc(self):
        r"""
        Poisson ratio. Following Mayne :cite:`mayne_2007`.
        """

        # assign size to poisson
        self.data.poisson = np.zeros(len(self.data.lithology))

        for i, lithology_index in enumerate(self.data.lithology):
            # if soft layer
            if lithology_index in ["1", "2", "3"]:
                self.data.poisson[i] = 0.5
            elif lithology_index == "4":
                self.data.poisson[i] = 0.25
            elif lithology_index in ["5", "6", "7"]:
                self.data.poisson[i] = 0.3
            else:
                self.data.poisson[i] = 0.375

    def permeability_calc(self):
        r"""
        Permeability calculation. Following Robertson :cite:`robertson_cabal_2014`.

        When  [$1.0 < I_{c} \leq 3.27$]

        .. math::

            k = 10^{0.952-3.04 I_{c}}

        When  [$3.27 < I_{c} < 4.0$]

        .. math::
            k = 10^{-4.52-1.37 I_{c}}
        """

        # assign size to permeability
        self.data.permeability = np.zeros(len(self.data.lithology))  # [m/s]

        for i, lit in enumerate(self.data.lithology):
            if self.data.IC[i] <= 3.27:
                self.data.permeability[i] = 10 ** (0.952 - 3.04 * self.data.IC[i])
            else:
                self.data.permeability[i] = 10 ** (-4.52 - 1.37 * self.data.IC[i])

    def qt_calc(self):
        r"""
        Corrected cone resistance, following Robertson and Cabal :cite:`robertson_cabal_2014`.

        .. math::

            q_{t} = q_{c} + u_{2} \left( 1 - a\right)
        """

        # qt computed following Robertson & Cabal (2015)
        # qt = qc + u2 * (1 - a)

        self.data.qt = self.data.tip + self.data.water * (1 - self.data.a)
        self.data.qt[self.data.qt <= 0] = 0

    def relative_density_calc(
        self,
        method: RelativeDensityMethod,
        c_0: Union[np.ndarray, float] = 15.7,
        c_2: Union[np.ndarray, float] = 2.41,
        Q_c: Union[np.ndarray, float] = 1,
        OCR: Union[np.ndarray, float] = 1,
        age: Union[np.ndarray, float] = 1000,
    ):
        r"""
        Computes relative density. Following methods described in Robertson :cite:`robertson_cabal_2014`. This method
        calculates the relative density for all the non cohesive soils along the whole cpt, i.e. RD is calculated when
        the lithology index is either 6, 7, 8 or 9.

        The relative density can be computed according to Baldi :cite:`baldi_1989` or
        Kulhawy and Mayne :cite:`kulhawy_1990`. Furthermore Kulhawy method can be simplified for most young,
        uncemented-based sands.

        .. math::

            RD_{Baldi} = (\frac{1}{C_{2}})LN(\frac{Q_{cn}}{C_{0}})

        .. math::

            RD_{Kulhawy}^{2} = \frac{Q_{cn}}{305 Q_{c} Q_{ocr} Q_{A}}

        .. math::

            Q_{ocr} = OCR^{0.18}

        .. math::

            Q_{A} = 1.2 + 0.05log(age/100)

        .. math::

            RD_{Kulhawy_simple}^{2} = \frac{Q_{tn}}{350}

        :param method: Method for calculation of relative density.
        :param c_0: (optional float or np array) soil constant for Baldi method. Default is 15.7
        :param c_2: (optional float or np array) soil constant for Baldi method. Default is 2.41
        :param Q_c: (optional float or np array) compressibility factor, 0.9 for low compressibility, 1.1 for
                    high compressibility. Default is 1.0
        :param OCR: (optional float or np array) Over consolidation ratio. Default = 1.0
        :param age: (optional float or np array) age of the soil in years. Default is 1000

        """

        self.data.relative_density = np.ones(len(self.data.qt)) * np.nan
        if method == RelativeDensityMethod.BALDI:
            # calculate normalised cpt resistance, corrected for overburden pressure
            Q_cn = (self.data.qt / self.data.Pa) / (
                self.data.effective_stress / self.data.Pa
            ) ** 0.5

            # calculate rd if Q_cn > c_0
            mask = Q_cn > c_0
            self.data.relative_density[mask] = (1 / c_2 * np.log(Q_cn / c_0))[mask]

        elif method == RelativeDensityMethod.KULHAWY:
            # calculate normalised cpt resistance, corrected for overburden pressure
            Q_cn = (self.data.qt / self.data.Pa) / (
                self.data.effective_stress / self.data.Pa
            ) ** 0.5

            # calculate overconsolidation factor
            Q_ocr = OCR ** 0.18

            # calculate aging factor
            Q_a = 1.2 + 0.05 * np.log10(age / 100)
            self.data.relative_density = np.sqrt(Q_cn / (305 * Q_c * Q_ocr * Q_a))

        elif method == RelativeDensityMethod.KULHAWY_SIMPLE:
            # method Kulhawy simple, valid for most young, uncemented silica based sands
            self.data.relative_density = np.sqrt(self.data.Qtn / 350)

        # check if soil is non cohesive
        is_non_cohesive = (
            (self.data.lithology == "6")
            + (self.data.lithology == "7")
            + (self.data.lithology == "8")
            + (self.data.lithology == "9")
        )

        # dispose_cohesive_soils. This order of removing the non cohesive soils is chosen, such that the above formulas
        # accept both floats and np arrays as inputs.
        self.data.relative_density[~is_non_cohesive] = np.nan

    def norm_cone_resistance_clean_sand_calc(self):
        """
        Calculates the clean sand equivalent normalised cone resistance, following Robertson and Cabal
        :cite:`robertson_cabal_2014`.

        .. math::

            Q_{tn,cs} = K_{c} \cdot Q_{tn}

        Where K_{c} is defined as follows:

        When  [$I_{c} \leq 1.64$]

        .. math::

            K_{c} = 1.0

        When  [$1.64 < I_{c} \leq 2.5$]

        .. math::
            K_{c} = 5.58 I_{c}^{3} - 0.403 I_{c}^{4} - 21.63 I_{c}^{2} + 33.65 I_{c} - 17.88

        When  [$1.64 < I_{c} <2.36$] and [$F_{r} < 0.5%$]

        .. math::
            K_{c} = 1.0

        When  [$2.5 < I_{c} <2.7$]

        .. math::
            K_{c} = 6 * 10^{-7} ( I_{c}^{16.76}
        """

        # initialise K_c and Q_tncs as nan
        K_c = np.ones(len(self.data.IC)) * np.nan
        self.data.Qtncs = np.ones(len(self.data.IC)) * np.nan

        # if IC is lower than 1.64, K_c is 1.0
        K_c[self.data.IC <= 1.64] = 1.0

        # calculate K_c for when (1.64 < IC <= 2.5)
        mask = (1.64 < self.data.IC) * (self.data.IC <= 2.5)
        K_c[mask] = (
            5.581 * self.data.IC[mask] ** 3
            - 0.403 * self.data.IC[mask] ** 4
            - 21.63 * self.data.IC[mask] ** 2
            + 33.75 * self.data.IC[mask]
            - 17.88
        )

        # calculate K_c for when (1.64 < IC <= 2.36) and Fr < 0.5
        mask = (1.64 < self.data.IC) * (self.data.IC <= 2.36) * (self.data.Fr < 0.5)
        K_c[mask] = 1.0

        # calculate K_c for when (2.5< IC <= 2.7)
        mask = (2.5 < self.data.IC) * (self.data.IC < 2.7)
        K_c[mask] = (6e-7) * self.data.IC[mask] ** 16.76

        # calculate Qtncs
        self.data.Qtncs = K_c * self.data.Qtn

    def state_parameter_calc(self):
        """
        Calculates state parameter from relationship with clean sand equivalent normalised cone resistance
        :cite:`robertson_2010`.

        .. math::

            \psi =0.56 - 0.33 log(Q_{tn,cs})
        """

        self.data.psi = 0.56 - 0.33 * np.log10(self.data.Qtncs)

    def filter(self, lithologies: List[str] = [""], key="", value: float = 0):
        r"""
        Filters the values of the CPT object.
        The filter removes the index of the object for the defined **lithologies**, where the **key** is smaller than
        the **value**
        The filter only removes the first consecutive samples.

        :param lithologies: list of lithologies to be filtered
        :param key: Key of the object to be filtered
        :param value: value of the key to be limited
        :return:
        """

        # if key is empty does nothing
        if not key:
            return

        # attributes to be changed
        attributes = [
            "depth",
            "depth_to_reference",
            "tip",
            "friction",
            "friction_nbr",
            "gamma",
            "rho",
            "total_stress",
            "effective_stress",
            "qt",
            "Qtn",
            "Fr",
            "IC",
            "n",
            "vs",
            "G0",
            "poisson",
            "damping",
            "water",
            "lithology",
            "litho_points",
            "inclination_resultant",
        ]

        # find indexes of the lithologies to be filtered
        idx_lito = []
        for lit in lithologies:
            idx_lito.extend(
                [i for i, val in enumerate(self.data.lithology) if val == lit]
            )

        # find indexes where the key attribute is smaller than the value
        idx_key = np.where(getattr(self, key) <= value)[0].tolist()

        # intercept indexes: append
        idx = list(set(idx_lito) & set(idx_key))

        # if nothing to delete : return
        if not idx:
            return

        # find first consecutive indexes
        idx_to_delete = [list(group) for group in mit.consecutive_groups(idx)][0]

        # check if zeros is in idx_to_delete: otherwise return
        if 0 not in idx_to_delete:
            return

        # delete all attributes
        for att in attributes:
            setattr(self, att, getattr(self, att)[idx_to_delete[-1] + 1 :])

        # correct depth
        self.data.depth -= self.data.depth[0]

    @staticmethod
    # TODO this function might be a duplicate
    def calculate_corrected_depth(penetration_length: Iterable, inclination: Iterable):
        r"""
        Correct the penetration length with the inclination angle

        :param penetration_length: measured penetration length
        :param inclination: measured inclination of the cone
        :return: corrected depth
        """
        corrected_d_depth = np.diff(penetration_length) * np.cos(
            np.radians(inclination[:-1])
        )
        corrected_depth = np.concatenate(
            (
                penetration_length[0],
                penetration_length[0] + np.cumsum(corrected_d_depth),
            ),
            axis=None,
        )
        return corrected_depth
