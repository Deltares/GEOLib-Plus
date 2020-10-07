"""Tools to read gef files"""
import os
import re
import numpy as np

def read_gef(gef_file, id, key_cpt, fct_a=0.8):

    index_dictionary = {'depth': None, 'tip': None, 'friction': None, 'friction_nb': None, 'pwp': None}
    dictionary_multiplication_factors = {'depth': 1., 'tip': 1000., 'friction': 1000., 'friction_nb': 1., 'pwp': 1000.}

    # read gef file
    with open(gef_file, 'r') as f:
        data = f.readlines()

    # search NAP
    idx_nap = [i for i, val in enumerate(data) if val.startswith(r'#ZID=')][0]
    NAP = float(data[idx_nap].split(',')[1])
    # search end of header
    idx_EOH = [i for i, val in enumerate(data) if val.startswith(r'#EOH=')][0]
    # # search for coordinates
    idx_coord = [i for i, val in enumerate(data) if val.startswith(r'#XYID=')][0]
    # search index depth
    for key_name in index_dictionary:
        index_dictionary[key_name] = read_column_index_for_gef_data(key_cpt[key_name], data)

    # read error
    idx_errors = [val.split(',')[1] for i, val in enumerate(data) if val.startswith(r'#COLUMNVOID=')]
    idx_errors_dict = match_idx_with_error(idx_errors, index_dictionary, dictionary_multiplication_factors)
    # rewrite data with separator ;
    data[idx_EOH + 1:] = [re.sub("[ :,!\t]+", ";", i.lstrip()) for i in data[idx_EOH + 1:]]

    try:
        # search index coefficient a
        idx_a = [i for i, val in enumerate(data) if val.endswith('Netto oppervlaktequotient van de conuspunt\n')][0]
        fct_a = float(data[idx_a].split(',')[1])
    except IndexError:
        fct_a = fct_a

    # remove empty lines
    data = list(filter(None, data))

    # read data & correct depth to NAP
    result_dictionary = read_data(index_dictionary, data, idx_EOH, dictionary_multiplication_factors)
    # if return dictionary is not a dictionary return false
    if not isinstance(result_dictionary, dict):
        return result_dictionary

    # if tip / friction / friction number are negative -> zero
    correct_for_negatives = ['tip', 'friction', 'friction_nb']
    correct_negatives_and_zeros(result_dictionary,correct_for_negatives)
    result_dictionary['pwp'] = np.array(result_dictionary['pwp'])
    # remove the points with error: value == -9999
    result_dictionary = remove_points_with_error(result_dictionary, idx_errors_dict)

    name = id
    coord = list(map(float, re.sub("[ ,!\t]+", ";", data[idx_coord].strip()).split("#XYID=")[-1].split(";")[2:4]))

    depth = [i for i in result_dictionary['depth']]
    z_NAP = [i - NAP for j, i in enumerate(result_dictionary['depth'])]

    res = dict(name=name,
               depth=np.array(depth),
               depth_to_reference=np.array(z_NAP),
               tip=np.array(result_dictionary['tip']),
               friction=np.array(result_dictionary['friction']),
               friction_nbr=np.array(result_dictionary['friction_nb']),
               a=fct_a,
               coordinates=coord,
               water=np.array(result_dictionary['pwp']),
               )
    return res

def read_column_index_for_gef_data(key_cpt, data):
    result = None
    for i, val in enumerate(data):
        if val.startswith(r'#COLUMNINFO=') and int(val.split(',')[-1]) == int(key_cpt):
            result = int(val.split(',')[0].split("=")[-1]) - 1
    return result

def match_idx_with_error(idx_errors, index_dictionary, dictionary_multiplycation_factors):
    index_error = dict(index_dictionary)
    # Make default in error dictionary
    for key in index_error:
        index_error[key] = None
    # Check if errors if not empty
    if bool(idx_errors):
        for key in index_dictionary:
            try:
                index_error[key] = float(idx_errors[int(index_dictionary[key])]) * dictionary_multiplycation_factors[key]
            except ValueError:
                index_error[key] = idx_errors[int(index_dictionary[key])]
            except TypeError:
                print(f'Warning: key {key} not found in GEF file')
                continue
    return index_error

def remove_points_with_error(result_dictionary, index_error):
    deleted_rows = 0
    for key in result_dictionary:
        for number, value in enumerate(result_dictionary[key]):
            if value == index_error[key]:
                result_dictionary = delete_value_for_all_keys(result_dictionary, number - deleted_rows)
                deleted_rows = deleted_rows + 1
    return result_dictionary

def delete_value_for_all_keys(result_dictionary, number):
    for key in result_dictionary:
        if isinstance(result_dictionary[key], list):
            del result_dictionary[key][number]
        elif isinstance(result_dictionary[key], np.ndarray):
            temp_list = result_dictionary[key].tolist()
            del temp_list[number]
            result_dictionary[key] = np.array(temp_list)
    return result_dictionary

def read_data(dictonary, data, idx_EOH, mult_factors):
    result_dictionary = dict(dictonary)
    for key in dictonary:
        try:
            result_dictionary[key] = [float(data[i].split(";")[dictonary[key]]) * mult_factors[key]
                                      for i in range(idx_EOH + 1, len(data))]
        except TypeError:
            # Pore pressures are not inputted
            if key == 'pwp':
                result_dictionary['pwp'] = np.zeros(len(result_dictionary['depth']))
            else:
                return f'CPT key: {key} not part of GEF file'

    return result_dictionary

def correct_negatives_and_zeros(result_dictionary, correct_for_negatives):
    for key in correct_for_negatives:
        result_dictionary[key] = np.array(result_dictionary[key])
        result_dictionary[key][result_dictionary[key] < 0] = 0
    return result_dictionary

def friction_ratio_calculation(friction, qt):
    # friction ratio calculation according to Robertson
    friction_ratio = np.array(friction) / np.array(qt) * 100.
    return friction_ratio

