from setuptools import setup, find_packages

import qics


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


setup(
    name='qics',
    version=qics.__version__,
    url=' ',
    license='Apache-2.0 License',
    author='-',
    author_email='-',
    description='-',
    long_description=readme(),
    packages=find_packages(),
    install_requires=[
        'numpy'
    ],
    zip_safe=True
)
