import ctypes
import os


def _load_libgauss():
    full_path = os.path.dirname(os.path.abspath(__file__))
    lib_path = "{}/lib/libgauss.so".format(full_path)
    lib = ctypes.cdll.LoadLibrary(lib_path)
    return lib


_libgauss = _load_libgauss()
_libgauss.gauss_vec_dot_f64.restype = ctypes.c_double


def _iterable_to_list(iterable):
    if isinstance(iterable, list):
        l = iterable
    else:
        l = list(iterable)
    return l


def _load_vec_f64(l):
    c_array = (ctypes.c_double * len(l))(*l)
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
            raise ValueError(
                "vectors not alligned for dot product, {} != {}".format(len(self), size)
            )

        if isinstance(b, Vec):
            b_vec = b
        else:
            b_vec = Vec(b)

        # TODO: detect datatype and call appropriate dot function
        result = _libgauss.gauss_vec_dot_f64(self._data, b_vec._data, size)
        return result


if __name__ == "__main__":
    pass
