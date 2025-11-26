from setuptools import setup, find_packages

LONG_DESCRIPTION="""
dRFEtools - A package for preforming dynamic recursive feature elimination with sklearn
=======================================================================

``dRFEtools`` is a package for dynamic recursive feature elimination supporting
random forest and several linear models for classification and regression.

Authors: Apuã Paquola, Kynon Jade Benjamin, and Tarun Katipalli

If using please cite: XXX.

Installation
============

``pip install --user dRFEtools``

"""

setup(
    name="dRFEtools",
    version="0.3.5",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.0.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "plotnine>=0.13.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.14.0",
        "statsmodels>=0.14.0",
    ],
    author="Kynon JM Benjamin",
    author_email="kj.benjamin90@gmail.com",
    description="A package for preforming dynamic recursive feature elimination with sklearn",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    package_data={
        "": ["*md"],
    },
    url="https://github.com/LieberInstitute/dRFEtools.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    keywords="random-forest recursive-feature-elimination sklearn linear-models feature-ranking",
    python_requires=">=3.10",
    zip_safe=False,
)
