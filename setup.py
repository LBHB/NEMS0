import codecs
import os.path
from setuptools import find_packages, setup

NAME = 'NEMS'

VERSION = '0.0.1a'

with codecs.open('README.md', encoding='utf-8') as f:
    long_description = f.read()

GENERAL_REQUIRES = ['numpy', 'scipy', 'matplotlib', 'pandas', 'requests',
                    'h5py', 'sqlalchemy']

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=True,
    author='LBHB',
    author_email='lbhb.ohsu@gmail.com',
    description='Neural Encoding Model System',
    long_description=long_description,
    url='http://neuralprediction.org',
    install_requires=GENERAL_REQUIRES,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    entry_points={
        'console_scripts': [
            'fit-model=scripts.fit_model:main',
            'download-demo-data=scripts.download_demo_data:main'
        ],
    }
)
