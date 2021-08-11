from typing import Dict, List, Optional
from geolib_plus.gef_cpt import GefCpt
from geolib_plus.cpt_base_model import AbstractInterpretationMethod, AbstractCPT

from tests.utils import TestUtils
import pytest
from pathlib import Path
from shapely import geometry
import shapefile
from matplotlib import pyplot as plt
import os
from pydantic import BaseModel


# Set up custom interpratation class
class CustomCptInterpretation(AbstractInterpretationMethod, BaseModel):
    cpt_data: AbstractCPT = None
    soil_types_for_classification: Dict = {}
    path_shapefile: Optional[Path] = None
    unit_weight_soil: List = []

    def interpret(self, cpt: AbstractCPT):
        """
        Function that interprets the cpt inputs.
        Lithology for each layer is determined according to
        the qc only method. Note that the pre_process method
        should be run before the interpret method.
        """
        # import cpt
        self.cpt_data = cpt
        # Perform unit tranformations
        self.cpt_data.friction = self.cpt_data.friction * 100
        self.cpt_data.friction[self.cpt_data.friction > 10] = 10

        sf = shapefile.Reader(str(self.path_shapefile))
        print("number of shapes imported:", len(sf.shapes()))
        plt.figure()
        for shape in list(sf.iterShapeRecords()):
            x_lon, y_lat = zip(*shape.shape.points)
            plt.fill(x_lon, y_lat, label=shape.record.name, alpha=0.4)
        plt.scatter(self.cpt_data.friction, self.cpt_data.tip)
        plt.xlabel("Friction ratio (Fr) [%]")
        plt.ylabel("CPT resistance (qc) [MPa]")
        plt.yscale("log")
        plt.legend()
        # plt.show()
        # read soil classification from shapefile
        self.soil_types()
        # calculate lithology
        self.lithology()
        # calculate unit weights based on the lithology found
        self.unit_weight()

    def unit_weight(self):
        """
        Function that determines the unit weight of different soil types depending
        on the classification type.
        """
        unit_weight = []
        typical_unit_weight_sand = 20
        typical_unit_weight_clay = 15
        typical_unit_weight_peat = 10
        for soil_type in self.cpt_data.lithology:
            if soil_type == "sand":
                unit_weight.append(typical_unit_weight_sand)
            elif soil_type == "clay":
                unit_weight.append(typical_unit_weight_clay)
            elif soil_type == "peat":
                unit_weight.append(typical_unit_weight_peat)
            else:
                unit_weight.append(None)
        self.unit_weight_soil = unit_weight

    def point_intersects_one_polygon(self, point):
        for soil_name, polygon in self.soil_types_for_classification.items():
            if point.intersects(polygon):
                return soil_name
        return None

    def lithology(self):
        """
        Function that reads a soil classification shapefile.
        """
        # determine into which soil type the point is
        lithology = []
        for counter in range(len(self.cpt_data.friction)):
            point_to_check = geometry.Point(
                self.cpt_data.tip[counter], self.cpt_data.friction[counter]
            )
            lithology.append(self.point_intersects_one_polygon(point_to_check))
        self.cpt_data.lithology = lithology

    def soil_types(self):
        r"""
        Function that read shapes from shape file and passes them as Polygons.

        :param path_shapefile: Path to the shapefile
        """

        # read shapefile
        sf = shapefile.Reader(str(self.path_shapefile))
        for polygon in list(sf.iterShapeRecords()):
            self.soil_types_for_classification[polygon.record.name] = geometry.Polygon(
                polygon.shape.points
            )


class TestCreateCustomInterpretation:
    @pytest.mark.integrationtest
    def test_create_custom_interpretation_class(self):
        # Create custom interpretation model based on the qc only rule as shapefile
        self.create_qc_based_rule()
        shapefile_location = Path(TestUtils.get_output_test_data_dir(""), "qc_rule")

        assert Path(str(shapefile_location) + ".shp").exists()
        # Read cpt using geolib +
        # open the gef file
        test_file = TestUtils.get_local_test_data_dir(
            Path("cpt", "gef", "CPT000000003688_IMBRO_A.gef")
        )
        assert test_file.is_file()
        # initialise models
        cpt = GefCpt()
        # test initial expectations
        assert cpt
        # read gef file
        cpt.read(filepath=test_file)
        # do pre-processing
        cpt.pre_process_data()
        # call custom interpretation class
        interpreter = CustomCptInterpretation()
        interpreter.path_shapefile = Path(str(shapefile_location) + ".shp")
        # use GEOLIB+ to run interpreter
        cpt.interpret_cpt(interpreter)
        os.remove(Path(str(shapefile_location) + ".dbf"))
        os.remove(Path(str(shapefile_location) + ".shp"))
        os.remove(Path(str(shapefile_location) + ".shx"))
        plt.figure()
        plt.plot(
            interpreter.unit_weight_soil,
            cpt.depth_to_reference,
            label=cpt.name,
        )
        plt.xlabel("Unit weight")
        plt.ylabel("depth")
        plt.legend()
        plt.show()

    def create_qc_based_rule(self):

        # define points
        A = geometry.Point(0, 0.1)
        B = geometry.Point(10, 0.1)
        C = geometry.Point(0, 1)
        D = geometry.Point(10, 1)
        E = geometry.Point(0, 6)
        F = geometry.Point(10, 6)
        G = geometry.Point(0, 1000)
        H = geometry.Point(10, 1000)
        # define polygons
        poligon_sand = geometry.Polygon([[p.x, p.y] for p in [E, G, H, F]])
        poligon_clay = geometry.Polygon([[p.x, p.y] for p in [E, F, D, C]])
        poligon_peat = geometry.Polygon([[p.x, p.y] for p in [C, D, B, A]])
        shapefile_polygon = {
            "sand": poligon_sand,
            "clay": poligon_clay,
            "peat": poligon_peat,
        }
        # write shapefile
        shapefile_location = Path(TestUtils.get_output_test_data_dir(""), "qc_rule")
        w = shapefile.Writer(shapefile_location)
        w.field("name", "C")
        final_list = []
        for key, value in shapefile_polygon.items():
            final_list.append(list(zip(value.exterior.xy[0], value.exterior.xy[1])))
            w.poly([list(zip(value.exterior.xy[0], value.exterior.xy[1]))])
            w.record(key)
        w.close()
