from setuptools import setup, find_packages

setup(
    name='amcl',
    version='0.0.1',
    url='https://github.com/BatsResearch/amcl.git',
    author='Dylan Sam, Alessio Mazzetto, Stephen Bach',
    author_email='dylan_sam@brown.edu, alessio_mazzetto@brown.edu, sbach@cs.brown.edu',
    description='Implementations of an adversarial multi-class learning approach from weak supervision sources',
    packages=find_packages(),
    install_requires=['numpy', 'sklearn', 'cvxpy', 'qpsolvers', 'pulp'],
)