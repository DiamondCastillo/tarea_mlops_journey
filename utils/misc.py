import os
import sys
import subprocess


def install_requirements(dir_requirements, name_requirements):

    path_requirements = os.path.join(dir_requirements, name_requirements)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', path_requirements])