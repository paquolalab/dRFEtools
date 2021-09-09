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

setup(name='dRFEtools',
      version='0.1.0',
      packages=find_packages(),
      install_requires=[
          'numpy>=1.20.1',
          'pandas>=1.2.4',
          'matplotlib>=3.3.4',
          'plotnine>=0.7.1',
          'scikit-learn>=0.24.1',
          'scipy>=1.6.0',
          'statsmodels>=0.12.2',
      ],
      author="Kynon JM Benjamin",
      author_email="kj.benjamin90@gmail.com",
      decription="A package for preforming dynamic recursive feature elimination with sklearn",
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/x-rst',
      package_data={
          '': ['*md'],
      },
      url="https://github.com/paquolalab/dRFEtools.git",
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
      ],
      keywords='random-forest recursive-feature-elimination sklearn linear-models feature-ranking',
      zip_safe=False)
