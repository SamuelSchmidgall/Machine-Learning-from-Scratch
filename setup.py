from setuptools import setup

setup(
    name='SciGen',
    author="Samuel Schmidgall",
    author_email="sschmidg@masonlive.gmu.edu",
    version='0.0.1',
    url="https://github.com/AbstractMobius/SciGen",
    packages=['scigen',],
    install_requires=['numpy'],
    license='MIT',
    long_description=open('README.md').read()
)