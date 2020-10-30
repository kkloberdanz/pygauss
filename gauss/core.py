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
_libgauss.gauss_simd_alloc.restype = ctypes.c_void_p
_libgauss.gauss_double_array_at.restype = ctypes.c_double
_libgauss.gauss_mean_double_array.restype = ctypes.c_double
_libgauss.gauss_median_double_array.restype = ctypes.c_double
_libgauss.gauss_variance_f64.restype = ctypes.c_double
_libgauss.gauss_standard_deviation_f64.restype = ctypes.c_double


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


def _alloc(nmemb):
    ptr = ctypes.c_void_p(_libgauss.gauss_simd_alloc(nmemb * 8))
    if not ptr:
        raise MemoryError("gauss could not allocate memory")
    else:
        return ptr
