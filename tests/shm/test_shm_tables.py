from geolib_plus.shm import shm_tables
import pytest


class TestShmTables:

    @pytest.mark.unittest
    def test_load_shm_tables(self):
        """
        Tests if values from table are correctly loaded and added to soils list
        """

        # get table
        table = shm_tables.ShmTables()
        table.load_shm_tables()

        # check if 24 soils are read
        assert len(table.soils) == 24

        # get first and last soil in list to check
        first_soil = table.soils[0]
        last_soil = table.soils[-1]

        # assert first soil
        assert first_soil.name == "Veen_mineraalarm"
        assert first_soil.unsaturated_weight.mean == 10.5
        assert first_soil.unsaturated_weight.limits == [10,11]

        assert first_soil.pop_layer.mean == 11
        assert first_soil.pop_layer.limits == [0, 60]
        assert first_soil.pop_layer.low_characteristic_value == 1

        # assert last soil
        assert last_soil.name == "Dijksmateriaal_base"
        assert last_soil.unsaturated_weight.mean == 17.5
        assert last_soil.unsaturated_weight.limits == [19, 23]

        assert last_soil.pop_layer.mean == 30
        assert last_soil.pop_layer.limits == [0, 150]
        assert last_soil.pop_layer.low_characteristic_value == 7
