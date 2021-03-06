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
_libgauss.gauss_vec_l1norm_f64.restype = ctypes.c_double
_libgauss.gauss_vec_l2norm_f64.restype = ctypes.c_double
_libgauss.gauss_vec_sum_f64.restype = ctypes.c_double
_libgauss.gauss_vec_index_max_f64.restype = ctypes.c_size_t
_libgauss.gauss_alloc.restype = ctypes.c_void_p
_libgauss.gauss_median_double_array.restype = ctypes.c_double
_libgauss.gauss_get_dtype.restype = ctypes.c_char_p
_libgauss.gauss_error_to_string.restype = ctypes.c_char_p


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


_dtype_to_gauss_type = {
    "prefered": -1,
    "float": 1,
    "double": 2,
    "cl_float": 3,
}

_dtype_to_ctype = {
    "double": ctypes.c_double,
    "float": ctypes.c_float,
    "cl_float": ctypes.c_float,
}


def _alloc(nmemb, dtype="double"):
    kind = _dtype_to_gauss_type[dtype]
    ptr = ctypes.c_void_p(_libgauss.gauss_alloc(nmemb, kind))
    if not ptr:
        raise MemoryError("gauss could not allocate memory")
    else:
        return ptr


def _free(ptr):
    _libgauss.gauss_free(ptr)


def _get_ctype(obj):
    return _dtype_to_ctype[_libgauss.gauss_get_dtype(obj).decode()]


def _error_to_string(e):
    return _libgauss.gauss_error_to_string(e).decode()
