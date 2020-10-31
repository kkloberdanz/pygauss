import ctypes
from . import _core

def mean_squared_error(y_true, y_predicted):
    l_true = len(y_true)
    l_pred = len(y_predicted)
    if (l_true != l_pred):
        msg = ("x and y vectors are not aligned: {} != {}"
            .format(l_true, l_pred))
        raise ValueError(msg)

    mse = ctypes.c_double(0.0)
    size = l_true
    error_code = _core._libgauss.guass_mean_squared_error(
        y_true._data, y_predicted._data, size, ctypes.byref(mse)
    )
    if error_code == -1:
        raise MemoryError("gauss failed to allocate memory")
    return mse.value
