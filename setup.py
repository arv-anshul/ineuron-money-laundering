from setuptools import find_packages, setup

requirements_txt = 'requirements.txt'
REMOVE_PACKAGE = '-e .'


def get_requirements() -> list[str]:
    with open(requirements_txt) as f:
        req_list = f.readline()
    req_list = [i.replace('\n', '') for i in req_list if i]

    if REMOVE_PACKAGE in req_list:
        req_list.remove(REMOVE_PACKAGE)
    return req_list


setup(
    name='Money Laundering',
    version='0.3.0',
    description="Money Laundering Prevention System",
    author='Anshul Raj Verma',
    author_email='arv.anshul.1864@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)
