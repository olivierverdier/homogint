from typing import Callable
from numpy.typing import NDArray
import numpy as np

def time_step(dt: np.float64) -> Callable[[Callable[[NDArray[np.float64]], NDArray[np.float64]]], Callable[[NDArray[np.float64]], NDArray[np.float64]]]:
    def scale(vf: Callable[[NDArray[np.float64]], NDArray[np.float64]]) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        def scaled_vf(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return dt*vf(x)
        return scaled_vf
    return scale
