from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='benchbot_eval',
      version='2.1.0',
      author='Ben Talbot',
      author_email='b.talbot@qut.edu.au',
      description=
      'The BenchBot evaluation tools for use with the BenchBot software stack',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=['numpy', 'shapely', 'benchbot_addons'],
      classifiers=(
          "Programming Language :: Python :: 2",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.3",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ))
