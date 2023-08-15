from setuptools import find_packages, setup

__author__ = 'Anshul Raj Verma'
__email__ = 'arv.anshul.1864@gmail.com'
__github__ = 'https://github.com/arv-anshul'


def get_requirements() -> list[str]:
    with open('requirements.txt') as f:
        return [i.strip('\n') for i in f.readline() if i != '-e .']


setup(
    name='Money Laundering',
    version='1.0.0',
    description="Money Laundering Prevention System",
    author=__author__,
    author_email=__email__,
    packages=find_packages(),
    install_requires=get_requirements(),
)
