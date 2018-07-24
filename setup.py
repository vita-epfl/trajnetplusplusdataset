"""setup trajnetdataset"""

from setuptools import setup

# extract version from __init__.py
with open('trajnetdataset/__init__.py', 'r') as f:
    VERSION_LINE = [l for l in f if l.startswith('__version__')][0]
    VERSION = VERSION_LINE.split('=')[1].strip()[1:-1]


setup(
    name='trajnetdataset',
    version=VERSION,
    packages=[
        'trajnetdataset',
    ],
    license='MIT',
    description='Trajnet dataset.',
    long_description=open('README.rst').read(),
    author='Sven Kreiss',
    author_email='me@svenkreiss.com',
    url='https://github.com/vita-epfl/trajnetdataset',

    install_requires=[
        'pysparkling',
        'scipy',
        'trajnettools',
    ],
    extras_require={
        'test': [
            'pylint',
            'pytest',
        ],
        'plot': [
            'matplotlib',
        ]
    },
)
