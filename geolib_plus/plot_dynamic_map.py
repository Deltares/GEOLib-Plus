"""
To use take for example an arleady read cpt by the geolib-plus library



Usage::

    >>> import geolib_plus
    >>> from geolib_plus.plot_dynamic_map import ObjectLocationMap, Location
    >>> from pathlib import Path
    >>> cpt_file_gef = Path("cpt", "gef", "test_cpt.gef")
    >>> cpt_gef = geolib_plus.gef_cpt.GefCpt()
    >>> cpt_gef.read(cpt_file_gef)
    >>> locations = [Location(x=cpt_gef.coordinates[0], y=cpt_gef.coordinates[1], label=cpt_gef.name, meta_data={'Norm': cpt_gef.cpt_standard})]
    >>> extract_map = ObjectLocationMap(object_locations=locations, results_folder=Path(""))

"""

import pathlib
from pathlib import Path
from typing import List

import folium
import pyproj
from pydantic import BaseModel


def number_div_icon(number):
    """
    Create a 'numbered' icon

    """
    icon = folium.features.DivIcon(
        icon_size=(90, 90),
        icon_anchor=(-15, 40),
        html='<div style="font-size: 9pt; font-weight: bold;text-shadow: white 1px 1px; align:left, color : black">'
        + "{:s}".format(number)
        + "</div>",
    )
    return icon


class Location(BaseModel):
    """
    Class that specifies the location of an object along with its metadata.

    :param x: Location x coordinate of the object in the crs 28992
    :param y: Location y coordinate of the object in the crs 28992
    :param label: label of the object
    :param meta_data: Meta data of the object

    """

    x: float
    y: float
    label: str
    meta_data: dict


class ObjectLocationMap(BaseModel):
    """
    Class that creates the cpt location map

    :param object_locations: List of locations of objects
    :param results_folder: Folder where the locations maps will be added

    """

    object_locations: List[Location]
    results_folder: Path

    def plot(self):
        """
        Function that creates an html map
        """
        self.plot_html_folium()

    @staticmethod
    def meta_data_string_for_label(meta_data):
        """
        Meta data are turned into html format
        """
        output_string = ""
        for key, value in meta_data.items():
            output_string = output_string + f"{key}: {value} <br>"
        return output_string

    def plot_html_folium(self):
        """
        Function that creates html map using folium package
        """
        transformer = pyproj.Transformer.from_crs(28992, 4326, always_xy=True)
        lon, lat = transformer.transform(
            self.object_locations[0].x, self.object_locations[0].y
        )
        m = folium.Map(location=[lat, lon], zoom_start=15, control_scale=True)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Esri Satellite",
            overlay=False,
            control=True,
        ).add_to(m)
        for location in self.object_locations:
            lon, lat = transformer.transform(location.x, location.y)
            label = (
                f"Bro id: {location.label} <br> "
                + ObjectLocationMap.meta_data_string_for_label(location.meta_data)
            )

            iframe = folium.IFrame(label, width=250, height=160)

            popup = folium.Popup(iframe)
            marker = folium.Marker(
                location=[lat, lon], popup=popup, tooltip=f"Bro id: {location.label}"
            )
            marker.add_to(m)
        feature_group = folium.FeatureGroup("labels")
        for location in self.object_locations:
            lon, lat = transformer.transform(location.x, location.y)
            label = f"{location.label}"
            folium.Marker(
                location=[lat, lon],
                icon=number_div_icon(label),
                draggable=True,
            ).add_to(feature_group)
        feature_group.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        m.save(str(pathlib.Path(self.results_folder, "dynamic_map_with_cpts.html")))
