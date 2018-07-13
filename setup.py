from setuptools import setup

# extract version from __init__.py
with open('trajnetdataset/__init__.py', 'r') as f:
    version_line = [l for l in f if l.startswith('__version__')][0]
    VERSION = version_line.split('=')[1].strip()[1:-1]


setup(
    name='trajnetdataset',
    version=VERSION,
    packages=[
        'trajnetdataset',
    ],
    license='MIT',
    description='Trajnet dataset.',
    long_description=open('README.md').read(),
    author='Sven Kreiss',
    author_email='me@svenkreiss.com',
    url='https://github.com/vita-epfl/trajnetdataset',

    install_requires=[
        'pysparkling',
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
