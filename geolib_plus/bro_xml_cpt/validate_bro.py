import logging
from urllib.parse import urlparse

import numpy as np
import requests
from lxml import etree

from geolib_plus.cpt_base_model import AbstractCPT


class HTTPSResolver(etree.Resolver):
    __name__ = "HTTPSResolver"

    def resolve(self, url, id_, context):
        url_components = urlparse(url)
        scheme = url_components.scheme

        # the resolver will only check for redirects if http/s is present
        if scheme == "http" or scheme == "https":
            head_response = requests.head(url, allow_redirects=True, verify=False)
            new_request_url = head_response.url
            if len(head_response.history) != 0:
                # recursively, resolve the ultimate redirection target
                return self.resolve(new_request_url, id, context)
            else:
                if scheme == "http":
                    # libxml2 can handle this resource
                    return self.resolve_filename(new_request_url, context)
                elif scheme == "https":
                    # libxml2 cannot resolve this resource, so do the work
                    get_response = requests.get(new_request_url, verify=False)
                    return self.resolve_string(
                        get_response.content, context, base_url=new_request_url
                    )
                else:
                    raise ValueError(f"{__name__}, Scheme should be http or https")
        else:
            # treat resource as a plain old file
            return self.resolve_filename(url, context)


def validate_bro(schema_url, bro_xml_file):
    bro_tree = etree.parse(bro_xml_file).find(
        "{http://www.broservices.nl/xsd/dscpt/1.1}dispatchDocument"
    )[0]
    parser = etree.XMLParser(load_dtd=True)
    resolver = HTTPSResolver(schema_url)
    parser.resolvers.add(resolver)
    xml_validator = etree.XMLSchema(etree.parse(schema_url, parser=parser))

    if xml_validator.validate(bro_tree):
        return 0
    else:
        xml_validator.assertValid(bro_tree)


def validate_bro_cpt(bro_xml_file):
    schema_url = r"https://schema.broservices.nl/xsd/dscpt/1.1/dscpt-messages.xsd"
    return validate_bro(schema_url, str(bro_xml_file))


class ValidatePreProcessing:
    def __check_file_contains_data(self, cpt: AbstractCPT):
        if len(cpt.penetration_length) == 0:
            logging.warning("File " + cpt.name + " contains no data")
            return

    def __check_data_different_than_zero(self, cpt: AbstractCPT):
        keys = [
            "penetration_length",
            "tip",
            "friction",
            "friction_ratio",
        ]
        for k in keys:
            if all(np.array(getattr(cpt, k)) == 0) or all(
                np.array(getattr(cpt, k)) is None
            ):
                logging.warning("File " + cpt.name + " contains empty data")
                return

    def __check_criteria_minimum_length(self, cpt, minimum_length: int):
        if np.max(np.abs(cpt.penetration_length)) < minimum_length:
            logging.warning(
                "File " + cpt.name + " has a length smaller than " + str(minimum_length)
            )
            return

    def __check_minimum_sample_criteria(self, cpt: AbstractCPT, minimum_samples: int):
        if len(cpt.penetration_length) < minimum_samples:
            logging.warning(
                "File "
                + cpt.name
                + " has a number of samples smaller than "
                + str(minimum_samples)
            )
            return

    def validate_length_and_samples_cpt(
        self, cpt: AbstractCPT, minimum_length: int = 5, minimum_samples: int = 50
    ):
        """
        Performs initial checks regarding the availability of
        data in the cpt. Returns a string that contains all the
        error messages.
        """

        self.__check_file_contains_data(cpt)
        self.__check_data_different_than_zero(cpt)
        self.__check_criteria_minimum_length(cpt, minimum_length)
        self.__check_minimum_sample_criteria(cpt, minimum_samples)
        return
