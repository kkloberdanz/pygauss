import ctypes
from . import _core


def ordinary_least_squares(x, y):
    """
    Ordinary Least Squares

    Uses optimized C implimentation. Results should be equivalent to the
    Python code below

    def ordinary_least_squares(x, y):
        mean_x = x.mean()
        mean_y = y.mean()
        m = (((x - mean_x) * (y - mean_y)).sum() /
             ((x - mean_x).square()).sum())
        b = mean_y - m * mean_x
        return m, b
    """
    lx = len(x)
    ly = len(y)
    if lx != ly:
        msg = "x and y vectors are not aligned: {} != {}".format(lx, ly)
        raise ValueError(msg)

    m = ctypes.c_double(0.0)
    b = ctypes.c_double(0.0)
    size = ctypes.c_size_t(lx)
    error_code = _core._libgauss.ordinary_least_squares(
        x._data, y._data, size, ctypes.byref(m), ctypes.byref(b)
    )
    if error_code == -1:
        raise MemoryError("gauss failed to allocate memory")
    return m.value, b.value
