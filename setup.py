from setuptools import find_packages, setup
from typing import List

Hypen =  '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This particular method will return the list of requirements

    '''

    requirements = []

    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n',' ') for req in requirements]

        if Hypen in requirements:
            requirements.remove(Hypen)

        
    return requirements

setup(
    name="Sales Predictor",
    version = '0.0.1',
    author = "Fakhar Abdullah",
    author_email='mywebdev3534@gmail.com',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')

)