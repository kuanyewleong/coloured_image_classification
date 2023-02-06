from pip._internal.req import parse_requirements
from setuptools import find_packages, setup

REQUIREMENTS = [str(req.requirement) for req in parse_requirements(
    'requirements.txt', session=None)]

if __name__ == "__main__":
    setup(
        name="pretrain_cr_model",
        use_scm_version=True,
        packages=find_packages("pretrain_cr_model", "pretrain_cr_model.*"),
        install_requires=REQUIREMENTS,
        setup_requires=['setuptools_scm'],
        url="https://github.com/ais-research/pretrain_cr_model",
        python_requires=">= 3.5",
    )