from lxml import etree
from urllib.parse import urlparse
import requests

class HTTPSResolver(etree.Resolver):
    __name__ = 'HTTPSResolver'

    def resolve(self, url, id_, context):
        url_components = urlparse(url)
        scheme = url_components.scheme

        # the resolver will only check for redirects if http/s is present
        if scheme == 'http' or scheme == 'https':
            head_response = requests.head(url, allow_redirects=True)
            new_request_url = head_response.url
            if len(head_response.history) != 0:
                # recursively, resolve the ultimate redirection target
                return self.resolve(new_request_url, id, context)
            else:
                if scheme == 'http':
                    # libxml2 can handle this resource
                    return self.resolve_filename(new_request_url, context)
                elif scheme == 'https':
                    # libxml2 cannot resolve this resource, so do the work
                    get_response = requests.get(new_request_url)
                    return self.resolve_string(get_response.content, context,
                                                base_url=new_request_url)
                else:
                    raise ValueError(f"{__name__}, Scheme should be http or https")
        else:
            # treat resource as a plain old file
            return self.resolve_filename(url, context)

def validate_bro(schema_url, bro_xml_file):
    bro_tree = etree.parse(bro_xml_file).find('{http://www.broservices.nl/xsd/dscpt/1.1}dispatchDocument')[0]
    parser = etree.XMLParser(load_dtd=True)
    resolver = HTTPSResolver(schema_url)
    parser.resolvers.add(resolver)
    xml_validator = etree.XMLSchema(etree.parse(schema_url, parser=parser))

    if xml_validator.validate(bro_tree):
        return 0
    else:
        xml_validator.assertValid(bro_tree)

def validate_bro_cpt(bro_xml_file):
    schema_url = r'https://schema.broservices.nl/xsd/dscpt/1.1/dscpt-messages.xsd'
    return validate_bro(schema_url, str(bro_xml_file))






