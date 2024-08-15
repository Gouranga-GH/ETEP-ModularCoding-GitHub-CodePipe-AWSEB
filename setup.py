# Import necessary modules from setuptools and typing
from setuptools import find_packages, setup
from typing import List

# Define a constant string to represent the editable installation flag
HYPEN_E_DOT = '-e .'

# Define a function to get the list of requirements from a file
def get_requirements(file_path: str) -> List[str]:
    '''
    This function reads a file containing a list of requirements and returns a list of those requirements.
    It removes the '-e .' entry if present.
    '''
    # Initialize an empty list to store requirements
    requirements = []
    
    # Open the file specified by file_path
    with open(file_path) as file_obj:
        # Read all lines from the file and store them in requirements
        requirements = file_obj.readlines()
        # Remove newline characters from each requirement
        requirements = [req.replace("\n", "") for req in requirements]
        
        # If the editable installation flag is in the requirements list, remove it
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    # Return the cleaned list of requirements
    return requirements

# Setup function for packaging the project
setup(
    name='ETEP-ModularCoding-GitHubActions-AWS',  # Name of the package
    version='0.0.1',  # Version of the package
    author='Gouranga',  # Author of the package
    author_email='post.gourang@gmail.com',  # Author's email address
    packages=find_packages(),  # Automatically find and include all packages in the project
    install_requires=get_requirements('requirements.txt')  # List of dependencies specified in requirements.txt
)
