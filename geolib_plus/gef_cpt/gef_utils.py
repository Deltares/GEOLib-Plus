"""Tools to read gef files."""
import re
import numpy as np
from typing import List, Dict
from pydantic import BaseModel


class GEFIndexValues(BaseModel):
    depth: int = 1
    tip: int = 2
    friction: int = 3
    friction_nb: int = 4
    pwp: int = 6


def read_gef(gef_file, id, key_cpt, fct_a=0.8):

    index_dictionary = {
        "depth": None,
        "tip": None,
        "friction": None,
        "friction_nb": None,
        "pwp": None,
    }
    dictionary_multiplication_factors = {
        "depth": 1.0,
        "tip": 1000.0,
        "friction": 1000.0,
        "friction_nb": 1.0,
        "pwp": 1000.0,
    }

    # read gef file
    with open(gef_file, "r") as f:
        data = f.readlines()

    # search NAP
    idx_nap = [i for i, val in enumerate(data) if val.startswith(r"#ZID=")][0]
    NAP = float(data[idx_nap].split(",")[1])
    # search end of header
    idx_EOH = [i for i, val in enumerate(data) if val.startswith(r"#EOH=")][0]
    # # search for coordinates
    idx_coord = [i for i, val in enumerate(data) if val.startswith(r"#XYID=")][0]
    # search index depth
    for key_name in index_dictionary:
        index_dictionary[key_name] = read_column_index_for_gef_data(
            key_cpt[key_name], data
        )

    # read error codes
    idx_errors = [
        val.split(",")[1]
        for i, val in enumerate(data)
        if val.startswith(r"#COLUMNVOID=")
    ]
    idx_errors_dict = match_idx_with_error(
        idx_errors, index_dictionary, dictionary_multiplication_factors
    )
    # rewrite data with separator ;
    data[idx_EOH + 1 :] = [
        re.sub("[ :,!\t]+", ";", i.lstrip()) for i in data[idx_EOH + 1 :]
    ]

    try:
        # search index coefficient a
        idx_a = [
            i
            for i, val in enumerate(data)
            if val.endswith("Netto oppervlaktequotient van de conuspunt\n")
        ][0]
        fct_a = float(data[idx_a].split(",")[1])
    except IndexError:
        fct_a = fct_a

    # remove empty lines
    data = list(filter(None, data))

    # read data & correct depth to NAP
    result_dictionary = read_data(
        index_dictionary, data, idx_EOH, dictionary_multiplication_factors
    )
    # if return dictionary is not a dictionary return false
    if not isinstance(result_dictionary, dict):
        return result_dictionary

    # if tip / friction / friction number are negative -> zero
    correct_for_negatives = ["tip", "friction", "friction_nb"]
    correct_negatives_and_zeros(result_dictionary, correct_for_negatives)
    result_dictionary["pwp"] = np.array(result_dictionary["pwp"])
    # remove the points with error: value == -9999
    result_dictionary = remove_points_with_error(result_dictionary, idx_errors_dict)

    name = id
    coord = list(
        map(
            float,
            re.sub("[ ,!\t]+", ";", data[idx_coord].strip())
            .split("#XYID=")[-1]
            .split(";")[2:4],
        )
    )

    depth = [i for i in result_dictionary["depth"]]
    z_NAP = [NAP - i for j, i in enumerate(result_dictionary["depth"])]

    res = dict(
        name=name,
        depth=np.array(depth),
        depth_to_reference=np.array(z_NAP),
        tip=np.array(result_dictionary["tip"]),
        friction=np.array(result_dictionary["friction"]),
        friction_nbr=np.array(result_dictionary["friction_nb"]),
        a=fct_a,
        coordinates=coord,
        water=np.array(result_dictionary["pwp"]),
    )
    return res


def read_column_index_for_gef_data(key_cpt: int, data: List[str]):
    """ In the gef file '#COLUMNINFO=id , name , column_number' format is used.
        This function returns the id number. Which will be later used 
        as reference for the errors.
        """
    # TODO I am not sure if this should raise an error or return None.
    # Will have to ask Bruno.
    result = None
    for i, val in enumerate(data):
        if val.startswith(r"#COLUMNINFO=") and int(val.split(",")[-1]) == int(key_cpt):
            result = int(val.split(",")[0].split("=")[-1]) - 1
    return result


def match_idx_with_error(
    idx_errors: List[str],
    index_dictionary: Dict,
    dictionary_multiplication_factors: Dict,
):
    """
    In the gef file each of the parameters has a value that is written
    when an error in the cpt data accumulation ocurred.
    """
    index_error = dict(index_dictionary)
    # Make default in error dictionary
    for key in index_error:
        index_error[key] = None
    # Check if errors if not empty
    if bool(idx_errors):
        for key in index_dictionary:
            try:
                index_error[key] = (
                    float(idx_errors[int(index_dictionary[key])])
                    * dictionary_multiplication_factors[key]
                )
            except ValueError:
                # Raises a ValueError at a string is returned
                index_error[key] = idx_errors[int(index_dictionary[key])]
            except IndexError:
                raise Exception(f"Key {key} not found in GEF file")
    return index_error


def remove_points_with_error(result_dictionary: Dict, index_error: Dict) -> Dict:
    """
    Values that contain data with errors should be removed
    from the resulting dictionary
    """
    deleted_rows = 0
    for key in result_dictionary:
        for number, value in enumerate(result_dictionary[key]):
            if value == index_error[key]:
                result_dictionary = delete_value_for_all_keys(
                    result_dictionary, number - deleted_rows
                )
                deleted_rows = deleted_rows + 1
    return result_dictionary


def delete_value_for_all_keys(result_dictionary: Dict, number: int) -> Dict:
    """
    Deletes index of all lists contained in the dictionary.
    """
    try:
        for key in result_dictionary:
            if isinstance(result_dictionary[key], list):
                del result_dictionary[key][number]
            elif isinstance(result_dictionary[key], np.ndarray):
                temp_list = result_dictionary[key].tolist()
                del temp_list[number]
                result_dictionary[key] = np.array(temp_list)
    except IndexError:
        raise Exception(f"Index <{number}> excides the length of list of key '{key}'")
    return result_dictionary


def read_data(
    input_dictionary: Dict, data: List[str], idx_EOH: int, mult_factors: Dict
) -> Dict:
    """
    Read column data from the gef file table.
    """
    result_dictionary = input_dictionary.copy()
    for key in input_dictionary:
        if key == "pwp" and not (input_dictionary["pwp"]):
            # Pore pressures are not inputted
            result_dictionary["pwp"] = np.zeros(len(result_dictionary["depth"]))
        else:
            try:
                result_dictionary[key] = [
                    float(data[i].split(";")[input_dictionary[key]]) * mult_factors[key]
                    for i in range(idx_EOH + 1, len(data))
                ]
            except TypeError:
                raise Exception(f"CPT key: {key} not part of GEF file")
    return result_dictionary


def correct_negatives_and_zeros(
    result_dictionary: Dict, correct_for_negatives: List[str]
) -> Dict:
    """
    Values tip / friction / friction cannot be negative so they 
    have to be zero.
    """
    for key in correct_for_negatives:
        result_dictionary[key] = np.array(result_dictionary[key])
        result_dictionary[key][result_dictionary[key] < 0] = 0
    return result_dictionary


def friction_ratio_calculation(friction, qt):
    # friction ratio calculation according to Robertson
    friction_ratio = np.array(friction) / np.array(qt) * 100.0
    return friction_ratio

