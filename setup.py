from setuptools import setup, find_packages

VERSION = '1.0.1'
DESCRIPTION = 'My first Python package'
LONG_DESCRIPTION = 'My first Python package with a slightly longer description'

# Setting up
setup(
    name="color_classify",
    version=VERSION,
    author="Supraja Chittari",
    author_email="suprajac@email.unc.edu",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url='https://github.com/UNC-Knight-Lab/ColorClassify',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'xlsxwriter',
        'scikit-learn',
        'scipy',
        'imbalanced-learn',
        'tk',
        'scikit-image',
        'Pillow'
    ],
    entry_points={
        'console_scripts': ['image_quant=color_classify.image_quant.command_line:main', 'bead_classification=color_classify.bead_class.command_line:main'],
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
    ],
    license='MIT'
)
