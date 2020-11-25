from geolib_plus.cpt_base_model import AbstractCPT, CptReader
from .gef_file_reader import GefFileReader

from pandas import DataFrame
from numpy import array, unique


class GefCpt(AbstractCPT):

    error_codes = {}

    @classmethod
    def get_cpt_reader(cls) -> CptReader:
        return GefFileReader()

    def remove_points_with_error(self):
        """
        Updates fields by removing depths that contain values with errors. i.e. incomplete data
        """
        # transform to dataframe
        update_dict = {}
        altered_keys = []
        for key in self.error_codes.keys():
            # ignore None values
            current_attribute = getattr(self, key)
            # also check length
            if current_attribute is None:
                continue
            else:
                altered_keys.append(key)
                keys = list(update_dict.keys())
                if len(keys) != 0 and current_attribute.size != len(
                        update_dict[keys[0]]
                ):
                    raise ValueError(
                        f"The data '{key}' (length = {current_attribute.size}) "
                        f"is not of the assumed data length = {len(update_dict[keys[0]])}"
                    )
                update_dict[key] = current_attribute

        update_dict = DataFrame.from_dict(update_dict)

        # remove points with error codes
        for key in self.error_codes.keys():
            if key in altered_keys:
                update_dict = update_dict[update_dict[key] != self.error_codes[key]]

        update_dict.to_dict("list")

        # update changed values in cpt
        for value in altered_keys:
            setattr(self, value, array(update_dict.get(value)))

    def has_points_with_error(self) -> bool:
        """
        A routine which checks whether the data is free of points with error.

        :return: If the gef cpt data is free of error flags
        """
        for key in self.error_codes.keys():
            current_attribute = getattr(self, key)
            if self.error_codes[key] in current_attribute:
                return True
        return False

    def drop_duplicate_depth_values(self):
        """
        Updates fields by removing penetration_length that are duplicate.
        """
        # TODO maybe here it makes more sense for the user to define what should not be duplicate

        # transform to dataframe
        update_dict = {}
        altered_keys = []
        for key in self.error_codes.keys():
            # ignore None values
            current_attribute = getattr(self, key)
            # also check length
            if current_attribute is None:
                continue
            else:
                altered_keys.append(key)
                keys = list(update_dict.keys())
                if len(keys) != 0 and len(current_attribute) != len(
                        update_dict[keys[0]]
                ):
                    raise ValueError(
                        f"The data '{key}' is not of the assumed data length = {len(update_dict[keys[0]])}"
                    )
                update_dict[key] = current_attribute

        update_dict = DataFrame.from_dict(update_dict)

        # perform action
        update_dict = update_dict.drop_duplicates(
            subset="penetration_length", keep="first"
        ).to_dict("list")

        # update changed values in cpt
        for value in altered_keys:
            setattr(self, value, array(update_dict.get(value)))

    def has_duplicated_depth_values(self) -> bool:
        """
        Check to see if there are any duplicate depth positions in the data
        :return True if has duplicate depths points based on penetration length
        """
        return len(unique(self.penetration_length)) != len(self.penetration_length)

