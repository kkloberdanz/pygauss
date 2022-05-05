# pygauss

Python bindings for GAUSS -- General Algorithmic Unified Statistical Solvers

## Build

pygauss depends on [gauss](https://github.com/kkloberdanz/gauss). First checkout
gauss, then run the following:

```
$ cd gauss/
$ make -j`nproc`
$ ls -l libgauss.so
-rwxrwxr-x 1 kyle kyle 38688 May  4 20:38 libgauss.so
$ cd ../pygauss/
$ cp ../gauss/libgauss.so gauss/lib/libgauss.so
```

Once `libgauss.so` is installed, you can then build the wheel for pygauss.

```
$ ./build.sh
$ python3 -m pip install dist/gauss-0.0.9-py3-none-any.whl
```

Gauss went through a heavy refactoring to enable GPU compute on the backend,
and functions that return vectors have not yet been implemented, however the
following methods below still work, and dot products are now being done on the
GPU using OpenCL.

Most of the compute is done in either OpenCL or OpenBLAS, but some algorithms
have been hand implemented using SIMD intrinsics.

```
$ python3
>>> import gauss
>>> a = gauss.Vec([1,2,3], dtype='double')
>>> a
Vec([1, 2, 3])
>>> a.l2norm()
3.7416573867739413
>>> a.l1norm()
6.0
>>> a.norm()
3.7416573867739413
>>> a.sum()
6.0
>>> a.argmax()
2
>>> a.variance()
0.6666666666666666
>>> a.dot(a)
14.0
>>>
```
