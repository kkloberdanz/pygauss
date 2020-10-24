import ctypes
import os
import atexit


def _load_libgauss():
    full_path = os.path.dirname(os.path.abspath(__file__))
    lib_path = "{}/lib/libgauss.so".format(full_path)
    lib = ctypes.cdll.LoadLibrary(lib_path)
    lib.gauss_init()
    return lib


_libgauss = _load_libgauss()
_libgauss.gauss_vec_dot_f64.restype = ctypes.c_double


def _exit_handler():
    global _libgauss
    _libgauss.gauss_close()


atexit.register(_exit_handler)


def _iterable_to_list(iterable):
    if isinstance(iterable, list):
        pylist = iterable
    else:
        pylist = list(iterable)
    return pylist


def _load_vec_f64(pylist):
    c_array = (ctypes.c_double * len(pylist))(*pylist)
    return c_array


def _load_vec_f32(pylist):
    c_array = (ctypes.c_float * len(pylist))(*pylist)
    return c_array


def _load_vec_i32(pylist):
    c_array = (ctypes.c_int32 * len(pylist))(*pylist)
    return c_array


def _load_vec_i64(pylist):
    c_array = (ctypes.c_int64 * len(pylist))(*pylist)
    return c_array


def _load_vec_u32(pylist):
    c_array = (ctypes.c_uint32 * len(pylist))(*pylist)
    return c_array


def _load_vec_u64(pylist):
    c_array = (ctypes.c_uint64 * len(pylist))(*pylist)
    return c_array


class Vec:
    def __init__(self, iterable=None):
        if iterable is not None:
            self._py_data = _iterable_to_list(iterable)

            # TODO: detect datatype and load it appropriately
            self._data = _load_vec_f64(self._py_data)

    def __len__(self):
        return len(self._py_data)

    def __repr__(self):
        return "Vec({})".format(repr(self._py_data))

    def dot(self, b):
        """Calculate the dot product of self and vector b"""
        size = len(b)
        if size != len(self):
            msg = "vectors not alligned for dot product, {} != {}".format(
                len(self), size
            )
            raise ValueError(msg)

        if isinstance(b, Vec):
            b_vec = b
        else:
            b_vec = Vec(b)

        # TODO: detect datatype and call appropriate dot function
        result = _libgauss.gauss_vec_dot_f64(self._data, b_vec._data, size)
        return result


if __name__ == "__main__":
    pass
