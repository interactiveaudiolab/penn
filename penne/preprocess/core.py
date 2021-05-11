"""core.py - data preprocessing"""


import penne
import torchaudio
import numpy as np
import tqdm
import json
import torch

###############################################################################
# Preprocess
###############################################################################


def dataset(dataset, voiceonly=False):
    """Preprocess dataset in data directory and save in cache

    Arguments
        name - string
            The name of the dataset to preprocess
    """
    subfolder = 'voiceonly' if voiceonly else 'all'
    output_directory = penne.CACHE_DIR / subfolder / dataset
    for subdir in ['annotation', 'audio', 'frames']:
        sub_directory = output_directory / subdir
        sub_directory.mkdir(exist_ok=True, parents=True)

    if dataset == 'MDB':
        MDB_process(output_directory, voiceonly)
    elif dataset == 'PTDB':
        PTDB_process(output_directory, voiceonly)
    else:
        raise ValueError(f'Dataset {dataset} is not implemented')


###############################################################################
# Dataset-specific
###############################################################################

def MDB_process(output_dir, voiceonly):
    offsets = {"totals": {}, "train": {}, "valid": {}, "test": {}}
    stem_dict = penne.data.partitions('MDB')

    for part in stem_dict.keys():
        total_frames = 0
        stems = stem_dict[part]

        for stem in tqdm.tqdm(stems, dynamic_ncols=True, desc=part):
            # Preprocess annotation
            annotation_file = penne.data.stem_to_annotation('MDB', stem)
            annotation = penne.load.MDB_pitch(annotation_file)
            if voiceonly:
                voiced = (annotation != 0).squeeze()
                annotation = annotation[:,voiced]
            offsets[part][penne.data.file_to_stem('MDB', annotation_file)] = [total_frames, annotation.shape[1]]
            total_frames += annotation.shape[1]
            filename = output_dir / "annotation" / f'{annotation_file.stem}.npy'
            np.save(filename, annotation)

            # Preprocess audio
            audio_file = penne.data.stem_to_file('MDB', stem)
            audio, sr = penne.load.audio(audio_file)
            audio = penne.resample(audio, sr)
            filename = output_dir / "audio" / audio_file.name
            torchaudio.save(filename, audio, penne.SAMPLE_RATE)

            pad_audio = torch.nn.functional.pad(audio, (penne.WINDOW_SIZE//2, penne.WINDOW_SIZE//2))
            frames = torch.nn.functional.unfold(
                    pad_audio[:, None, None, :],
                    kernel_size=(1, penne.WINDOW_SIZE),
                    stride=(1, penne.HOP_SIZE))
            if voiceonly:
                frames = frames[:,:,voiced]
            frame_filename = output_dir / "frames" / f'{audio_file.stem}.npy'
            np.save(frame_filename, frames)

        offsets["totals"][part] = total_frames

    with open(output_dir / "offsets.json", 'w') as f:
        json.dump(offsets, f, indent=4)


def PTDB_process(output_dir, voiceonly):
    offsets = {"totals": {}, "train": {}, "valid": {}, "test": {}}
    stem_dict = penne.data.partitions('PTDB')

    for part in stem_dict.keys():
        total_frames = 0
        stems = stem_dict[part]

        for stem in tqdm.tqdm(stems, dynamic_ncols=True, desc=part):
            # Preprocess annotation
            annotation_file = penne.data.stem_to_annotation('PTDB', stem)
            annotation = penne.load.PTDB_pitch(annotation_file)
            if voiceonly:
                voiced = (annotation != 0).squeeze()
                annotation = annotation[:,voiced]
            offsets[part][penne.data.file_to_stem('PTDB', annotation_file)] = [total_frames, annotation.shape[1]]
            total_frames += annotation.shape[1]
            filename = output_dir / "annotation" / f'{annotation_file.stem}.npy'
            np.save(filename, annotation)

            # Preprocess audio
            audio_file = penne.data.stem_to_file('PTDB', stem)
            audio, sr = penne.load.audio(audio_file)
            audio = penne.resample(audio, sr)
            filename = output_dir / "audio" / audio_file.name
            torchaudio.save(filename, audio, penne.SAMPLE_RATE)

            frames = torch.nn.functional.unfold(
                    audio[:, None, None, :],
                    kernel_size=(1, penne.WINDOW_SIZE),
                    stride=(1, penne.HOP_SIZE))
            if voiceonly:
                frames = frames[:,:,voiced]
            frame_filename = output_dir / "frames" / f'{audio_file.stem}.npy'
            np.save(frame_filename, frames)

        offsets["totals"][part] = total_frames

    with open(output_dir / "offsets.json", 'w') as f:
        json.dump(offsets, f, indent=4)