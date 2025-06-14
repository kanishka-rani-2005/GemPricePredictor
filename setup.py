from setuptools import find_namespace_packages,setup
from typing import List

def get_requirements(file_path:str)->List[str]:

    requires=[]
    with open(file_path, 'r') as f:
        requires=f.readlines()


        requires=[r.replace('\n','') for r in requires]

        if '-e .' in requires:
            requires.remove('-e .')
    return requires


setup(
    name='my_package',
    version='1.0',
    author='Kanishka Rani',
    author_email='kanishka22043@gmail.com',
    packages=find_namespace_packages(),
    install_requires=get_requirements('requirements.txt')
)



