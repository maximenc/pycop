from setuptools import setup, find_packages

setup(
    name = 'pycop',
    version = '0.0.10',  
    description = 'Copula for multivariate dependence modeling',
    long_description=open('README.md', 'r').read(),
    long_description_content_type="text/markdown",
    author = 'Maxime N',
    author_email = 'maxime.nlc@proton.me',
    url = 'https://github.com/maximenc/pycop/',
    download_url = 'https://github.com/maximenc/pycop/',
    classifiers = [],
    include_package_data=True,
    packages=find_packages(".")
)