"""core.py - data preprocessing"""


import penne
import torchaudio
import numpy as np
import tqdm
import json

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
    total_frames = 0
    offsets = {"stems": {}}
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
        offsets["stems"][penne.data.file_to_stem('MDB', annotation_file)] = [total_frames, annotation.shape[1]]
        total_frames += annotation.shape[1]
        filename = output_dir / "annotation" / f'{annotation_file.stem}.npy'
        np.save(filename, annotation)

    offsets["total"] = total_frames
    with open(output_dir / "offsets.json", 'w') as f:
        json.dump(offsets, f, indent=4)


def PTDB_process(output_dir):
    total_frames = 0
    offsets = {"stems": {}}
    stems = penne.partition.PTDB_stems()
    files = [penne.data.stem_to_file('PTDB', stem) for stem in stems]
    # for audio_file in tqdm.tqdm(files, dynamic_ncols=True, desc="Audio"):
    #     audio, sr = penne.load.audio(audio_file)
    #     audio = penne.resample(audio, sr)
    #     filename = output_dir / "audio" / audio_file.name
    #     torchaudio.save(filename, audio, penne.SAMPLE_RATE)
    
    annotation_files = [penne.data.stem_to_annotation('PTDB', stem) for stem in stems]
    for annotation_file in tqdm.tqdm(annotation_files, dynamic_ncols=True, desc="Annotations"):
        annotation = penne.load.PTDB_pitch(annotation_file)
        offsets["stems"][penne.data.file_to_stem('PTDB', annotation_file)] = [total_frames, annotation.shape[1]]
        total_frames += annotation.shape[1]
        filename = output_dir / "annotation" / f'{annotation_file.stem}.npy'
        np.save(filename, annotation)

    offsets["total"] = total_frames
    with open(output_dir / "offsets.json", 'w') as f:
        json.dump(offsets, f, indent=4)