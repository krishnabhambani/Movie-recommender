from setuptools import setup

with open("Readme.md","r", encoding="utf-8") as fh:
    long_description=fh.read()

AUTHOR_NAME= ' Krishna Bhambani '
SRC_REPO= 'src'
LIST_OF_REQUIREMENTS = ['streamlit']

setup(
    name= SRC_REPO,
    version= '0.0.1',
    author= AUTHOR_NAME
)