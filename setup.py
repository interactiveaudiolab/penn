from setuptools import find_packages, setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='penn',
    description='Pitch Estimating Neural Networks (PENN)',
    version='0.0.14',
    author='Max Morrison, Caedon Hsieh, Nathan Pruyne, and Bryan Pardo',
    author_email='interactiveaudiolab@gmail.com',
    url='https://github.com/interactiveaudiolab/penn',
    extras_require={
        'train': [
            'librosa',     # 0.9.1
            'matplotlib',  # 3.6.1
            'pyworld',     # 0.3.2
            'scipy',       # 1.9.3
            'torchcrepe'   # 0.0.17
        ]
    },
    install_requires=[
        'huggingface_hub', # 0.11.1
        'numpy',           # 1.23.4
        'torch',           # 1.12.1+cu113
        'torchaudio',      # 0.12.1+cu113
        'torchutil',       # 0.0.7
        'yapecs'           # 0.0.6
    ],
    packages=find_packages(),
    package_data={'penn': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['audio', 'frequency', 'music', 'periodicity', 'pitch', 'speech'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
