import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gauss",
    version="0.0.1",
    author="Kyle Kloberdanz",
    author_email="kyle.g.kloberdanz@gmail.com",
    description="Python bindings for Gauss (General Algorithmic Unified Statistical Solvers)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kkloberdanz/pygauss",
    packages=setuptools.find_packages(),
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
