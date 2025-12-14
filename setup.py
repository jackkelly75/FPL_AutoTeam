from setuptools import setup, find_packages

setup(
    name="FPL_AutoTeam",
    version="0.1.0",
    description="Predict Fantasy Premier League points and transfers using fixtures, data imports, and simulations.",
    author="JKelly",
    author_email="research.foundation.ai@gmail.com",
    url="https://github.com/jackkelly75/FPL_AutoTeam",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],  # leave empty if you manage deps via environment.yml
)