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
_libgauss.gauss_vec_norm_f64.restype = ctypes.c_double
_libgauss.gauss_vec_sumabs_f64.restype = ctypes.c_double
_libgauss.gauss_vec_index_max_f64.restype = ctypes.c_size_t
_libgauss.gauss_simd_alloc.restype = ctypes.c_void_p
_libgauss.gauss_double_array_at.restype = ctypes.c_double


def _exit_handler():
    # global _libgauss
    # _libgauss.gauss_close()
    pass


atexit.register(_exit_handler)


def _iterable_to_list(iterable):
    if isinstance(iterable, list):
        pylist = iterable
    else:
        pylist = list(iterable)
    return pylist


#def _load_vec_f64(pylist):
#    c_array = (ctypes.c_double * len(pylist))(*pylist)
#    return c_array
#
#
#def _load_vec_f32(pylist):
#    c_array = (ctypes.c_float * len(pylist))(*pylist)
#    return c_array
#
#
#def _load_vec_i32(pylist):
#    c_array = (ctypes.c_int32 * len(pylist))(*pylist)
#    return c_array
#
#
#def _load_vec_i64(pylist):
#    c_array = (ctypes.c_int64 * len(pylist))(*pylist)
#    return c_array
#
#
#def _load_vec_u32(pylist):
#    c_array = (ctypes.c_uint32 * len(pylist))(*pylist)
#    return c_array
#
#
#def _load_vec_u64(pylist):
#    c_array = (ctypes.c_uint64 * len(pylist))(*pylist)
#    return c_array


def _setup_binop(self, other):
    if isinstance(other, _number_types):
        raise ValueError("vector scaling not yet implemented")
    elif isinstance(other, Vec):
        b = other
    else:
        b = Vec(other)

    size = len(b)
    if size != len(self):
        s1 = len(self)
        s2 = len(b)
        msg = "vectors not alligned for add, {} != {}".format(s1, s2)
        raise ValueError(msg)
    buf = ctypes.c_void_p(_libgauss.gauss_simd_alloc(size * 8))
    dst = Vec(frompointer=(buf, size))
    return dst, b


_number_types = (int, float)


class Vec:
    def __init__(self, iterable=None, dtype=None, frompointer=None):
        self._data = None
        if iterable is not None:
            # TODO: detect datatype and load it appropriately
            pydata = _iterable_to_list(iterable)
            self._len = len(pydata)
            self._data = ctypes.c_void_p(_libgauss.gauss_simd_alloc(self._len * 8))
            for i in range(self._len):
                value = ctypes.c_double(pydata[i])
                _libgauss.gauss_set_double_array_at(self._data, i, value)
        elif frompointer:
            ptr, nmemb = frompointer
            self._data = ptr
            self._len = nmemb

    def __del__(self):
        if self._data is not None:
            _libgauss.gauss_free(self._data);
            self._data = None

    def __len__(self):
        return self._len

    def __repr__(self):
        if len(self) < 10:
            pydata = list(self)
            return "Vec({})".format(repr(pydata))
        else:
            start = ', '.join(str(self[x]) for x in range(5))
            end = ', '.join(str(self[x]) for x in range(len(self) - 5, len(self)))
            return "Vec([{}, ..., {}])".format(start, end)

    def __getitem__(self, index):
        if index >= self._len:
            raise StopIteration
        else:
            return _libgauss.gauss_double_array_at(self._data, index)

    def __add__(self, other):
        dst, b = _setup_binop(self, other)
        _libgauss.gauss_add_double_array(dst._data, self._data, b._data, len(self))
        return Vec(dst)

    def __floordiv__(self, other):
        dst, b = _setup_binop(self, other)
        _libgauss.gauss_floordiv_double_array(dst._data, self._data, b._data, len(self))
        return Vec(dst)

    def __truediv__(self, other):
        dst, b = _setup_binop(self, other)
        _libgauss.gauss_div_double_array(dst._data, self._data, b._data, len(self))
        return Vec(dst)

    def __mul__(self, other):
        dst, b = _setup_binop(self, other)
        _libgauss.gauss_mul_double_array(dst._data, self._data, b._data, len(self))
        return Vec(dst)

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

    def norm(self):
        return _libgauss.gauss_vec_norm_f64(self._data, len(self))

    def sumabs(self):
        return _libgauss.gauss_vec_sumabs_f64(self._data, len(self))

    def index_max(self):
        return _libgauss.gauss_vec_index_max_f64(self._data, len(self))

    def max(self):
        return self[self.index_max()]

    def sqrt(self):
        dst = _libgauss.gauss_simd_alloc(len(self) * 8)
        _libgauss.gauss_sqrt_double_array(dst, self._data, len(self))
        ret = Vec(frompointer=(dst, len(self)))
        return ret


if __name__ == "__main__":
    pass
