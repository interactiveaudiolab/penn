from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='penne',
    description='Pitch Estimating Neural NEtworks (PENNE)',
    version='0.0.1',
    author='Max Morrison, Caedon Hsieh, Nathan Pruyne, and Bryan Pardo',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/interactiveaudiolab/penne',
    install_requires=[
        'numpy',       # 1.23.4
        'scipy',       # 1.9.3
        'torch',       # 1.12.1+cu113
        'tqdm',        # 4.64.1
        'torchaudio',  # 0.12.1+cu113
        'yapecs'       # TODO
    ],
    packages=['penne'],
    package_data={'penne': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['pitch', 'audio', 'speech', 'music', 'pytorch', 'crepe'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
