import ctypes
from . import _core


_number_types = (int, float)


def _setup_binop(self, other):
    if isinstance(other, _number_types):
        size = len(self)
        buf = _core._alloc(size)
        dst = Vec(frompointer=(buf, size))
        return dst, other
    elif isinstance(other, Vec):
        b = other
    else:
        b = Vec(other)

    size = len(b)
    if size != len(self):
        msg = "vectors not alligned for add, {} != {}".format(len(self), size)
        raise ValueError(msg)
    buf = _core._alloc(size)
    dst = Vec(frompointer=(buf, size))
    return dst, b


class Vec:
    """
    Gauss Vector

    Uses optimized native code on backend
    """

    def __init__(self, iterable=None, dtype=None, frompointer=None):
        self._data = None
        self._pydata = []
        self._dtype = dtype if dtype is not None else "prefered"
        if iterable is not None:
            # TODO: detect datatype and load it appropriately
            pydata = _core._iterable_to_list(iterable)
            self._pydata = pydata
            self._len = len(pydata)
            self._data = _core._alloc(self._len, self._dtype)
            if dtype in {"float", "cl_float"}:
                buf = (ctypes.c_float * self._len)(*pydata)
            else:
                buf = (ctypes.c_double * self._len)(*pydata)
            err = _core._libgauss.gauss_set_buffer(self._data, buf)
            if err != 0:
                raise Exception("failed to set gauss buffer")
        elif frompointer:
            ptr, nmemb = frompointer
            self._data = ptr
            self._len = nmemb

    def __del__(self):
        if self._data is not None:
            _core._free(self._data)
            self._data = None

    def __len__(self):
        return self._len

    def __repr__(self):
        if len(self) < 12:
            pydata = list(self)
            return "Vec({})".format(repr(pydata))
        else:
            start = ", ".join(str(self[x]) for x in range(5))
            end = ", ".join(
                str(self[x]) for x in range(len(self) - 5, len(self))
            )
            return "Vec([{}, ..., {}])".format(start, end)

    def __getitem__(self, index):
        if index >= self._len:
            raise StopIteration
        else:
            return self._pydata[index]

    #    def __setitem__(self, index, item):
    #        if index >= self._len:
    #            raise IndexError
    #        else:
    #            value = ctypes.c_double(item)
    #            return _core._libgauss.gauss_set_double_array_at(
    #                self._data, index, value
    #            )

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        dst, b = _setup_binop(self, other)
        if isinstance(b, _number_types):
            value = ctypes.c_double(b)
            _core._libgauss.gauss_add_double_scalar(
                dst._data, self._data, value, len(self)
            )
        else:
            _core._libgauss.gauss_add_double_array(
                dst._data, self._data, b._data, len(self)
            )
        return dst

    def __sub__(self, other):
        dst, b = _setup_binop(self, other)
        if isinstance(b, _number_types):
            value = ctypes.c_double(b)
            _core._libgauss.gauss_sub_double_scalar(
                dst._data, self._data, value, len(self)
            )
        else:
            _core._libgauss.gauss_sub_double_array(
                dst._data, self._data, b._data, len(self)
            )
        return dst

    def __floordiv__(self, other):
        dst, b = _setup_binop(self, other)
        if isinstance(b, _number_types):
            value = ctypes.c_double(b)
            _core._libgauss.gauss_floordiv_double_scalar(
                dst._data, self._data, value, len(self)
            )
        else:
            _core._libgauss.gauss_floordiv_double_array(
                dst._data, self._data, b._data, len(self)
            )
        return dst

    def __truediv__(self, other):
        dst, b = _setup_binop(self, other)
        if isinstance(b, _number_types):
            value = ctypes.c_double(b)
            _core._libgauss.gauss_div_double_scalar(
                dst._data, self._data, value, len(self)
            )
        else:
            _core._libgauss.gauss_div_double_array(
                dst._data, self._data, b._data, len(self)
            )
        return dst

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        dst, b = _setup_binop(self, other)
        if isinstance(b, _number_types):
            value = ctypes.c_double(b)
            _core._libgauss.gauss_vec_scale_f64(
                dst._data, self._data, len(self), value
            )
        else:
            _core._libgauss.gauss_mul_double_array(
                dst._data, self._data, b._data, len(self)
            )
        return dst

    def dot(self, b):
        """Calculate the dot product of self and vector b"""
        size = len(b)
        if size <= 0:
            raise ValueError("dot on empty vector")
        if size != len(self):
            msg = "vectors not alligned for dot product, {} != {}".format(
                len(self), size
            )
            raise ValueError(msg)

        if isinstance(b, Vec):
            b_vec = b
        else:
            b_vec = Vec(b)

        if self._dtype in {"float", "cl_float"}:
            result = ctypes.c_float(0.0)
        else:
            result = ctypes.c_double(0.0)
        err = _core._libgauss.gauss_vec_dot(
            self._data, b_vec._data, ctypes.byref(result)
        )
        if err != 0:
            raise Exception("error calculating dot product: {}".format(_core._error_to_string(err)))
        return result.value

    def l1norm(self):
        """L1 norm, equivalent to sum of the absolute values of the vector"""
        if len(self) <= 0:
            raise ValueError("l1norm on empty vector")
        return _core._libgauss.gauss_vec_l1norm_f64(self._data, len(self))

    def l2norm(self):
        """L2 norm, also known as euclidean distance"""
        ctype = _core._get_ctype(self._data)
        result = ctype(4.2)
        err = _core._libgauss.gauss_vec_l2norm(
            self._data, ctypes.byref(result)
        )
        if err != 0:
            raise Exception("error calculating l2 norm")
        return result.value

    def norm(self):
        """Alias for L2 norm, also known as euclidean distance"""
        if len(self) <= 0:
            raise ValueError("norm on empty vector")
        return self.l2norm()

    def sum(self):
        """Sum of elements"""
        if len(self) <= 0:
            raise ValueError("sum on empty vector")
        if self._dtype in {"float", "cl_float"}:
            result = ctypes.c_float(0.0)
        else:
            result = ctypes.c_double(0.0)
        err = _core._libgauss.gauss_vec_sum(self._data, ctypes.byref(result))
        if err != 0:
            raise Exception("error calculating sum")
        return result.value

    def argmax(self):
        """Index of the maximum element"""
        if len(self) <= 0:
            raise ValueError("argmax on empty vector")
        return _core._libgauss.gauss_vec_index_max_f64(self._data, len(self))

    def max(self):
        """Maximum element"""
        if len(self) <= 0:
            raise ValueError("max on empty vector")
        return self[self.argmax()]

    def min(self):
        """Minimum element"""
        if len(self) <= 0:
            raise ValueError("min on empty vector")
        return _core._libgauss.gauss_min_vec_f64(self._data, len(self))

    def sqrt(self):
        """Element by element square root"""
        ptr = _core._alloc(len(self))
        dst = Vec(frompointer=(ptr, len(self)))
        _core._libgauss.gauss_sqrt_double_array(
            dst._data, self._data, len(self)
        )
        return dst

    def square(self):
        """Element by element square"""
        return self * self

    def mean(self):
        """Mean of vector"""
        if len(self) <= 0:
            raise ValueError("mean on empty vector")
        return _core._libgauss.gauss_mean_double_array(self._data, len(self))

    def median(self):
        """Median of vector"""
        if len(self) <= 0:
            raise ValueError("median on empty vector")
        scratch = _core._alloc(len(self) * 8)
        value = _core._libgauss.gauss_median_double_array(
            scratch, self._data, len(self)
        )
        _core._libgauss.gauss_free(scratch)
        return value

    def variance(self):
        """Variance of vector"""
        if len(self) <= 0:
            raise ValueError("varience on empty vector")
        return _core._libgauss.gauss_variance_f64(self._data, len(self))

    def standard_deviation(self):
        """Standard deviation of vector"""
        if len(self) <= 0:
            raise ValueError("standard_deviation on empty vector")
        return _core._libgauss.gauss_standard_deviation_f64(
            self._data, len(self)
        )


if __name__ == "__main__":
    pass
