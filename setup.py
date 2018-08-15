#-*- coding:utf8 -*-
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='jeteloss',  
    version='0.2',   
    author = "Long-Gang Pang, Ya-Yun He and Xin-Nian Wang",
    author_email = "lgpang@qq.com, heyayun@gmail.com, xnwang@lbl.gov",
    description = "Data-driven extraction of jet energy loss distributions in heavy-ion collisions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/snowhitiger/jeteloss",
    packages=find_packages('jeteloss'),
    include_package_data=True,
    exclude_package_date={'':['.gitignore']},
    install_requires=[
        'numpy>=1.14',
        'pymc>=2.3.6',
        'h5py>=2.8.0',
        'matplotlib>=2.2.0',
        'scipy>=1.1.0'
    ],
    license = "MIT",
    keywords = "Bayesian, MCMC, Jet energy loss extractor, RAA",
    classifiers=(
        #"Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
