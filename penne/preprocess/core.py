"""core.py - data preprocessing"""


import penne
import torchaudio
import numpy as np
import tqdm

###############################################################################
# Preprocess
###############################################################################


def dataset(dataset):
    """Preprocess dataset in data directory and save in cache

    Arguments
        name - string
            The name of the dataset to preprocess
    """

    output_directory = penne.CACHE_DIR / dataset

    if dataset == 'MDB':
        MDB_process(output_directory)
    elif dataset == 'PTDB':
        PTDB_process(output_directory)
    else:
        raise ValueError(f'Dataset {dataset} is not implemented')


###############################################################################
# Dataset-specific
###############################################################################

def MDB_process(output_dir):
    stems = penne.partition.MDB_stems()
    files = [penne.data.stem_to_file('MDB', stem) for stem in stems]
    for audio_file in tqdm.tqdm(files, dynamic_ncols=True, desc="Audio"):
        audio, sr = penne.load.audio(audio_file)
        audio = penne.resample(audio, sr)
        filename = output_dir / "audio" / audio_file.name
        torchaudio.save(filename, audio, penne.SAMPLE_RATE)
    
    annotation_files = [penne.data.stem_to_annotation('MDB', stem) for stem in stems]
    for annotation_file in tqdm.tqdm(annotation_files, dynamic_ncols=True, desc="Annotations"):
        annotation = penne.load.MDB_pitch(annotation_file)
        filename = output_dir / "annotation" / f'{annotation_file.stem}.npy'
        np.save(filename, annotation)

def PTDB_process(output_dir):
    stems = penne.partition.PTDB_stems()
    files = [penne.data.stem_to_file('PTDB', stem) for stem in stems]
    for audio_file in tqdm.tqdm(files, dynamic_ncols=True, desc="Audio"):
        audio, sr = penne.load.audio(audio_file)
        audio = penne.resample(audio, sr)
        filename = output_dir / "audio" / audio_file.name
        torchaudio.save(filename, audio, penne.SAMPLE_RATE)
    
    annotation_files = [penne.data.stem_to_annotation('PTDB', stem) for stem in stems]
    for annotation_file in tqdm.tqdm(annotation_files, dynamic_ncols=True, desc="Annotations"):
        annotation = penne.load.PTDB_pitch(annotation_file)
        filename = output_dir / "annotation" / f'{annotation_file.stem}.npy'
        np.save(filename, annotation)