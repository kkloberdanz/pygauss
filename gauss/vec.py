import ctypes
from . import core


_number_types = (int, float)


def _setup_binop(self, other):
    if isinstance(other, _number_types):
        size = len(self)
        buf = core._alloc(size)
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
    buf = core._alloc(size)
    dst = Vec(frompointer=(buf, size))
    return dst, b


class Vec:
    def __init__(self, iterable=None, dtype=None, frompointer=None):
        self._data = None
        if iterable is not None:
            # TODO: detect datatype and load it appropriately
            pydata = core._iterable_to_list(iterable)
            self._len = len(pydata)
            self._data = core._alloc(self._len)
            for i in range(self._len):
                value = ctypes.c_double(pydata[i])
                core._libgauss.gauss_set_double_array_at(self._data, i, value)
        elif frompointer:
            ptr, nmemb = frompointer
            self._data = ptr
            self._len = nmemb

    def __del__(self):
        if self._data is not None:
            core._libgauss.gauss_free(self._data)
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
            return core._libgauss.gauss_double_array_at(self._data, index)

    def __setitem__(self, index, item):
        if index >= self._len:
            raise IndexError
        else:
            value = ctypes.c_double(item)
            return core._libgauss.gauss_set_double_array_at(
                self._data, index, value
            )

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        dst, b = _setup_binop(self, other)
        if isinstance(b, _number_types):
            value = ctypes.c_double(b)
            core._libgauss.gauss_add_double_scalar(
                dst._data, self._data, value, len(self)
            )
        else:
            core._libgauss.gauss_add_double_array(
                dst._data, self._data, b._data, len(self)
            )
        return dst

    def __sub__(self, other):
        dst, b = _setup_binop(self, other)
        if isinstance(b, _number_types):
            value = ctypes.c_double(b)
            core._libgauss.gauss_sub_double_scalar(
                dst._data, self._data, value, len(self)
            )
        else:
            core._libgauss.gauss_sub_double_array(
                dst._data, self._data, b._data, len(self)
            )
        return dst

    def __floordiv__(self, other):
        dst, b = _setup_binop(self, other)
        if isinstance(b, _number_types):
            value = ctypes.c_double(b)
            core._libgauss.gauss_floordiv_double_scalar(
                dst._data, self._data, value, len(self)
            )
        else:
            core._libgauss.gauss_floordiv_double_array(
                dst._data, self._data, b._data, len(self)
            )
        return dst

    def __truediv__(self, other):
        dst, b = _setup_binop(self, other)
        if isinstance(b, _number_types):
            value = ctypes.c_double(b)
            core._libgauss.gauss_div_double_scalar(
                dst._data, self._data, value, len(self)
            )
        else:
            core._libgauss.gauss_div_double_array(
                dst._data, self._data, b._data, len(self)
            )
        return dst

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        dst, b = _setup_binop(self, other)
        if isinstance(b, _number_types):
            value = ctypes.c_double(b)
            core._libgauss.gauss_vec_scale_f64(
                dst._data, self._data, len(self), value
            )
        else:
            core._libgauss.gauss_mul_double_array(
                dst._data, self._data, b._data, len(self)
            )
        return dst

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
        result = core._libgauss.gauss_vec_dot_f64(self._data, b_vec._data, size)
        return result

    def l1norm(self):
        return core._libgauss.gauss_vec_l1norm_f64(self._data, len(self))

    def l2norm(self):
        return core._libgauss.gauss_vec_l2norm_f64(self._data, len(self))

    def norm(self):
        return self.l2norm()

    def sum(self):
        return core._libgauss.gauss_vec_sum_f64(self._data, len(self))

    def argmax(self):
        return core._libgauss.gauss_vec_index_max_f64(self._data, len(self))

    def max(self):
        return self[self.argmax()]

    def sqrt(self):
        ptr = core._alloc(len(self))
        dst = Vec(frompointer=(ptr, len(self)))
        core._libgauss.gauss_sqrt_double_array(dst._data, self._data, len(self))
        return dst

    def square(self):
        return self * self

    def mean(self):
        return core._libgauss.gauss_mean_double_array(self._data, len(self))

    def median(self):
        return core._libgauss.gauss_median_double_array(self._data, len(self))

    def variance(self):
        return core._libgauss.gauss_variance_f64(self._data, len(self))

    def standard_deviation(self):
        return core._libgauss.gauss_standard_deviation_f64(self._data, len(self))


if __name__ == "__main__":
    pass
