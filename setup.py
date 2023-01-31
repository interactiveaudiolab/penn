from setuptools import find_packages, setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='penn',
    description='Pitch Estimating Neural Networks (PENN)',
    version='0.0.3',
    author='Max Morrison, Caedon Hsieh, Nathan Pruyne, and Bryan Pardo',
    author_email='interactiveaudiolab@gmail.com',
    url='https://github.com/interactiveaudiolab/penn',
    install_requires=[
        'huggingface_hub', # 0.11.1
        'numpy',           # 1.23.4
        'tensorboard',     # 2.11.0
        'torch',           # 1.12.1+cu113
        'tqdm',            # 4.64.1
        'torchaudio',      # 0.12.1+cu113
        'yapecs'           # 0.0.6
    ],
    packages=find_packages(),
    package_data={'penn': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['audio', 'frequency', 'music', 'periodicity', 'pitch', 'speech'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
