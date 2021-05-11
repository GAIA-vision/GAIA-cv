""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path as osp


here = osp.abspath(osp.dirname(__file__))
with open(osp.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('gaiavision/version.py').read())


if __name__ == '__main__':
    setup(
        name='gaiavision',
        version=__version__,
        description='An AutoML transfer learning toolbox',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Junran Peng',
        author_email='jrpeng4ever@126.com',
        classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Education/Engineering/Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
        ],
        license='Apache License 2.0',
        keywords='computer vision, object detection, transfer learning, AutoML',
        packages=find_packages(exclude=('configs', 'tools', 'apps', 'scripts')),
        install_requires=[
            'torch >= 1.5.1',
            'torchvision',
            'pandas >= 1.1.5',
            'mmcv==1.2.7'],
        python_requires='>=3.6',
        zip_save=False,
    )
