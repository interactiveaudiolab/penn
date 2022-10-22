import shutil
import ssl
import tarfile
import urllib
import zipfile

import penne


###############################################################################
# Download datasets
###############################################################################


def datasets(datasets):
    """Download datasets"""
    if 'mdb' in datasets:
        mdb()

    if 'ptdb' in datasets:
        ptdb()


###############################################################################
# Individual datasets
###############################################################################


def mdb():
    """Download mdb dataset"""
    # Download
    url = 'https://zenodo.org/record/1481172/files/MDB-stem-synth.tar.gz'
    file = penne.SOURCE_DIR / 'mdb.tar.gz'
    download_file(url, file)

    with penne.data.chdir(penne.DATA_DIR):

        # Unzip
        with tarfile.open(file, 'r:gz') as tfile:
            tfile.extractall()

            # Rename directory
            shutil.move('MDB-stem-synth', 'mdb')


def ptdb():
    """Download ptdb dataset"""
    # Download
    url = 'https://www2.spsc.tugraz.at/databases/PTDB-TUG/SPEECH_DATA_ZIPPED.zip'
    file = penne.SOURCE_DIR / 'ptdb.zip'
    download_file(url, file)

    with penne.data.chdir(penne.DATA_DIR):

        # Unzip
        with zipfile.ZipFile(file, 'r') as zfile:
            zfile.extractall('ptdb')


###############################################################################
# Utilities
###############################################################################


def download_file(url, file):
    """Download file from url"""
    with urllib.request.urlopen(url, context=ssl.SSLContext()) as response, \
         open(file, 'wb') as output:
        shutil.copyfileobj(response, output)
