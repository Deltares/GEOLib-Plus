from pydantic import BaseModel

import numpy as np


class GeneralUtils(BaseModel):
    """
    Contains general static functions.
    """

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def linearise_data_over_layer(
        data_to_linearized: np.array, depth: np.array, buffer_zone: int = 5
    ):
        """
        Function that returns the linearised data of the layer inputted by the user.
        """
        if (len(data_to_linearized) <= buffer_zone * 2) or (
            len(depth) <= buffer_zone * 2
        ):
            raise ValueError(
                "The lenth of the arrays inputted are  smaller than the buffer zone. "
            )
        buffer_depth = abs(depth[buffer_zone] - depth[0])
        depth_of_layer = abs(depth[-1] - depth[0])
        if not (depth_of_layer > 2 * buffer_depth):
            raise ValueError(
                "The depth of the layer should be more than 2 time the depth of the buffer zone."
            )
        y = data_to_linearized[buffer_zone:-buffer_zone]
        x = depth[buffer_zone:-buffer_zone]
        a, b = np.polyfit(x, y, 1)
        new_output = a * depth + b
        return new_output
