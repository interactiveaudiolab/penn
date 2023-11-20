import shutil

import torchutil

import penn


###############################################################################
# Download datasets
###############################################################################


@torchutil.notify('download')
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
    torchutil.download.targz(
        'https://zenodo.org/record/1481172/files/MDB-stem-synth.tar.gz',
        penn.DATA_DIR)

    # Delete previous directory
    shutil.rmtree(penn.DATA_DIR / 'mdb', ignore_errors=True)

    # Rename directory
    shutil.move(penn.DATA_DIR / 'MDB-stem-synth', penn.DATA_DIR / 'mdb')


def ptdb():
    """Download ptdb dataset"""
    directory = penn.DATA_DIR / 'ptdb'
    directory.mkdir(exist_ok=True, parents=True)
    torchutil.download.zip(
        'https://www2.spsc.tugraz.at/databases/PTDB-TUG/SPEECH_DATA_ZIPPED.zip',
        directory)
