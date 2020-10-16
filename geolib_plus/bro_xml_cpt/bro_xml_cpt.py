# External modules
import numpy as np

from geolib_plus.cpt_base_model import AbstractCPT
from .bro_utils import xml_to_byte_string, parse_bro_xml
from .validate_bro import validate_bro_cpt


class BroXmlCpt(AbstractCPT):

    water_measurement_type = []
    __water_measurement_types = ["porePressureU1", "porePressureU2", "porePressureU3"]

    def read(self, bro_xml_file):

        # validate bro xml file
        validate_bro_cpt(bro_xml_file)

        # read the BRO_XML into Memory
        xml = xml_to_byte_string(bro_xml_file)

        # parse the BRO_XML to BRO CPT Dataset
        bro_cpt_dataset = parse_bro_xml(xml)

        # if BRO_XML is not a dictionary: returns error message
        if not isinstance(bro_cpt_dataset, dict):
            return bro_cpt_dataset  # Should there be an error message

        # add the BRO_XML attributes to CPT structure
        self.__parse_bro(bro_cpt_dataset)

    def __parse_bro(
        self,
        cpt,
        minimum_length=5,
        minimum_samples=50,
        minimum_ratio=0.1,
        convert_to_kPa=True,
    ):
        """
        Parse the BRO information into the object structure

        Parameters
        ----------
        :param cpt: BRO cpt dataset
        :param minimum_length: (optional) minimum length that cpt files needs to have
        :param minimum_samples: (optional) minimum samples that cpt files needs to have
        :param minimum_ratio: (optional) minimum ratio of positive values that cpt files needs to have
        :param convert_to_kPa: (optional) convert units to kPa
        :return:
        """

        # remove NAN row from the dataframe
        for key in cpt["dataframe"]:
            cpt["dataframe"] = cpt["dataframe"].dropna(subset=[key])

        # check if file contains data
        if len(cpt["dataframe"].penetrationLength) == 0:
            message = "File " + cpt["id"] + " contains no data"
            return message

        # check if data is different than zero:
        keys = ["penetrationLength", "coneResistance", "localFriction", "frictionRatio"]
        for k in keys:
            if all(cpt["dataframe"][k] == 0):
                message = "File " + cpt["id"] + " contains empty data"
                return message

        # parse cpt file name
        self.name = cpt["id"]
        # parse coordinates
        self.coordinates = [cpt["location_x"], cpt["location_y"]]

        # parse reference datum
        key = "vertical_datum"
        self.vertical_datum = cpt[key] if key in cpt else []

        # parse local reference point
        key = "local_reference"
        self.local_reference = cpt[key] if key in cpt else []

        # parse quality class
        key = "quality_class"
        self.quality_class = cpt[key] if key in cpt else []

        # parse cone penetrator type
        key = "cone_penetrometer_type"
        self.cpt_type = cpt[key] if key in cpt else []

        # parse cpt standard
        key = "cpt_standard"
        self.cpt_standard = cpt[key] if key in cpt else []

        # parse result time
        key = "result_time"
        self.result_time = cpt[key] if key in cpt else []

        # parse measurement type of pore pressure
        self.water_measurement_type = [
            water_measurement_type
            for water_measurement_type in self.__water_measurement_types
            if water_measurement_type in cpt["dataframe"]
        ]
        if not self.water_measurement_type:
            self.water_measurement_type = "no_measurements"
        else:
            self.water_measurement_type = self.water_measurement_type[0]

        # check criteria of minimum length
        if np.max(np.abs(cpt["dataframe"].penetrationLength.values)) < minimum_length:
            message = (
                "File "
                + cpt["id"]
                + " has a length smaller than "
                + str(minimum_length)
            )
            return message

        # check criteria of minimum samples
        if len(cpt["dataframe"].penetrationLength.values) < minimum_samples:
            message = (
                "File "
                + cpt["id"]
                + " has a number of samples smaller than "
                + str(minimum_samples)
            )
            return message

        # check data consistency: remove doubles depth
        cpt["dataframe"] = cpt["dataframe"].drop_duplicates(
            subset="penetrationLength", keep="first"
        )

        # check if there is a pre_drill. if so pad the data
        (
            depth,
            cone_resistance,
            friction_ratio,
            local_friction,
            pore_pressure,
        ) = self.__define_pre_drill(cpt, length_of_average_points=minimum_samples)

        # parse inclination resultant
        if "inclinationResultant" in cpt["dataframe"]:
            self.inclination_resultant = cpt["dataframe"]["inclinationResultant"].values
        else:
            self.inclination_resultant = np.empty(len(depth)) * np.nan

        # check quality of CPT
        # if more than minimum_ratio CPT is corrupted: discard CPT
        if (
            len(cone_resistance[cone_resistance <= 0]) / len(cone_resistance)
            > minimum_ratio
            or len(cone_resistance[local_friction <= 0]) / len(local_friction)
            > minimum_ratio
        ):
            message = "File " + cpt["id"] + " is corrupted"
            return message

        # unit in kPa is required for correlations
        unit_converter = 1000.0 if convert_to_kPa else 1.0

        # parse depth
        self.depth = depth
        # parse surface level
        self.local_reference_level = cpt["offset_z"]
        # parse NAP depth
        self.depth_to_reference = self.local_reference_level - depth
        # parse tip resistance
        self.tip = cone_resistance * unit_converter
        self.tip[self.tip <= 0] = 0.0
        # parse friction
        self.friction = local_friction * unit_converter
        self.friction[self.friction <= 0] = 0.0
        # parser friction number
        self.friction_nbr = friction_ratio
        self.friction_nbr[self.friction_nbr <= 0] = 0.0
        # read a
        self.a = cpt["a"]
        # default water is zero
        self.water = np.zeros(len(self.depth))
        # if water exists parse water
        if self.water_measurement_type in self.__water_measurement_types:
            self.water = pore_pressure * unit_converter

        return True

    def __define_pre_drill(self, cpt_BRO, length_of_average_points=3):
        r"""
        Checks the existence of pre-drill.
        If predrill exists it add the average value of tip, friction and friction number to the pre-drill length.
        The average is computed over the length_of_average_points.
        If pore water pressure is measured, the pwp is assumed to be zero at surface level.

        Parameters
        ----------
        :param cpt_BRO: BRO cpt dataset
        :param length_of_average_points: number of samples of the CPT to be used to fill pre-drill
        :return: depth, tip resistance, friction number, friction, pore water pressure
        """

        starting_depth = 0
        pore_pressure = None

        depth = self.__get_depth_from_bro(cpt_BRO)

        if float(cpt_BRO["predrilled_z"]) != 0.0:
            # if there is pre-dill add the average values to the pre-dill

            # Set the discretisation
            dicretisation = np.average(np.diff(depth))

            # find the average
            average_cone_res = np.average(
                cpt_BRO["dataframe"]["coneResistance"][:length_of_average_points]
            )
            average_fr_ratio = np.average(
                cpt_BRO["dataframe"]["frictionRatio"][:length_of_average_points]
            )
            average_loc_fr = np.average(
                cpt_BRO["dataframe"]["localFriction"][:length_of_average_points]
            )

            # Define all in the lists
            local_depth = np.arange(
                starting_depth, float(cpt_BRO["predrilled_z"]), dicretisation
            )
            local_cone_res = np.repeat(average_cone_res, len(local_depth))
            local_fr_ratio = np.repeat(average_fr_ratio, len(local_depth))
            local_loc_fr = np.repeat(average_loc_fr, len(local_depth))

            # if there is pore water pressure
            # Here the endpoint is False so that for the final of local_pore_pressure I don't end up with the same value
            # as the first in the Pore Pressure array.
            for water_measurement_type in self.__water_measurement_types:
                if water_measurement_type in cpt_BRO["dataframe"]:
                    local_pore_pressure = np.linspace(
                        0,
                        cpt_BRO["dataframe"][water_measurement_type].values[0],
                        len(local_depth),
                        endpoint=False,
                    )
                    pore_pressure = np.append(
                        local_pore_pressure,
                        cpt_BRO["dataframe"][water_measurement_type].values,
                    )

            # Enrich the Penetration Length
            depth = np.append(
                local_depth, local_depth[-1] + dicretisation + depth - depth[0]
            )
            coneresistance = np.append(
                local_cone_res, cpt_BRO["dataframe"]["coneResistance"].values
            )
            frictionratio = np.append(
                local_fr_ratio, cpt_BRO["dataframe"]["frictionRatio"].values
            )
            localfriction = np.append(
                local_loc_fr, cpt_BRO["dataframe"]["localFriction"].values
            )

        else:
            # No predrill existing: just parsing data
            depth = depth - depth[0]
            coneresistance = cpt_BRO["dataframe"]["coneResistance"].values
            frictionratio = cpt_BRO["dataframe"]["frictionRatio"].values
            localfriction = cpt_BRO["dataframe"]["localFriction"].values

            # if there is pore water pressure
            for water_measurement_type in self.__water_measurement_types:
                if water_measurement_type in cpt_BRO["dataframe"]:
                    pore_pressure = cpt_BRO["dataframe"][water_measurement_type].values

        # correct for missing samples in the top of the CPT
        if depth[0] > 0:
            # add zero
            depth = np.append(0, depth)
            coneresistance = np.append(
                np.average(
                    cpt_BRO["dataframe"]["coneResistance"][:length_of_average_points]
                ),
                coneresistance,
            )
            frictionratio = np.append(
                np.average(
                    cpt_BRO["dataframe"]["frictionRatio"][:length_of_average_points]
                ),
                frictionratio,
            )
            localfriction = np.append(
                np.average(
                    cpt_BRO["dataframe"]["localFriction"][:length_of_average_points]
                ),
                localfriction,
            )

            # if there is pore water pressure
            for water_measurement_type in self.__water_measurement_types:
                if water_measurement_type in cpt_BRO["dataframe"]:
                    pore_pressure = np.append(
                        np.average(
                            cpt_BRO["dataframe"][water_measurement_type][
                                :length_of_average_points
                            ]
                        ),
                        pore_pressure,
                    )

        return depth, coneresistance, frictionratio, localfriction, pore_pressure

    def __get_depth_from_bro(self, cpt_BRO):
        """
        If depth is present in the bro cpt and is valid, the depth is parsed from depth
        elseif resultant inclination angle is present and valid in the bro cpt, the penetration length is corrected with
        the inclination angle.
        if both depth and inclination angle are not present/valid, the depth is parsed from the penetration length.
        :param cpt_BRO: dataframe
        :return:
        """
        depth = np.array([])
        if "depth" in cpt_BRO["dataframe"]:
            if cpt_BRO["dataframe"]["depth"].values.dtype == np.dtype("float64"):
                depth = cpt_BRO["dataframe"]["depth"].values
        if "inclinationResultant" in cpt_BRO["dataframe"] and depth.size == 0:
            if cpt_BRO["dataframe"]["inclinationResultant"].values.dtype == np.dtype(
                "float64"
            ):
                depth = self.calculate_corrected_depth(
                    cpt_BRO["dataframe"]["penetrationLength"].values,
                    cpt_BRO["dataframe"]["inclinationResultant"].values,
                )
        if depth.size == 0:
            depth = cpt_BRO["dataframe"]["penetrationLength"].values
        return depth

