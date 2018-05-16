import os
import re
import setuptools
from pathlib import Path

p = Path(__file__)

def get_version(package):
    """
    Return package version as listed in `__version__` in `init.py`.
    """
    init_py = open(os.path.join(package, '__init__.py')).read()
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)

version = get_version('chrono_initialization')

setup_requires = [
    'numpy',
    'pytest-runner'
]

install_requires = [
    'scikit-learn',
    'tqdm',
]
test_require = [
    'pytest-cov',
    'pytest-html',
    'pytest',
    'coverage'
]

setuptools.setup(
    name="chrono_initialization",
    version=version,
    python_requires='>3.6',    
    author="Koji Ono",
    author_email="kbu94982@gmail.com",
    description="Chrono Initialization pytorch implementation",
    long_description=(p.parent / 'README.md').open(encoding='utf-8').read(),
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=test_require,
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)