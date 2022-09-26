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


def dataset(dataset, clean=False):
    """Preprocess dataset in data directory and save in cache

    Arguments
        name - string
            The name of the dataset to preprocess
    """
    # make directories if they do not exist
    all_dir = penne.CACHE_DIR / 'all' / dataset
    voiceonly_dir = penne.CACHE_DIR / 'voiceonly' / dataset
    audio_dir = penne.CACHE_DIR / 'audio' / dataset
    harmo_dir = penne.CACHE_DIR / 'harmo' / dataset
    for output_dir in [all_dir, voiceonly_dir, harmo_dir]:
        for subdir in ['annotation', 'frames']:
            sub_directory = output_dir / subdir
            sub_directory.mkdir(exist_ok=True, parents=True)
    audio_dir.mkdir(exist_ok=True, parents=True)

    if dataset in ['MDB', 'PTDB']:
        preprocess_data(dataset, all_dir, voiceonly_dir, audio_dir, harmo_dir, clean)
    else:
        raise ValueError(f'Dataset {dataset} is not implemented')

def preprocess_data(dataset, all_dir, voiceonly_dir, audio_dir, harmo_dir, clean):
    all_offsets = {"totals": {}, "train": {}, "valid": {}, "test": {}}
    voiceonly_offsets = {"totals": {}, "train": {}, "valid": {}, "test": {}}
    stem_dict = penne.data.partitions(dataset, clean)

    harmo_instance = penne.harmo_preprocess.PitchTracker(dataset=dataset)

    for part in stem_dict.keys():
        all_total_frames = 0
        voiceonly_total_frames = 0
        stems = stem_dict[part]

        for stem in tqdm.tqdm(stems, dynamic_ncols=True, desc=part):
            # PREPREPROCESS ANNOTATION

            # load annotation
            annotation_file = penne.data.stem_to_annotation(dataset, stem)
            annotation = penne.load.pitch_annotation(dataset, annotation_file)
            n_annotation_frames = annotation.shape[1]
                
            # save offset to start of stem
            all_offsets[part][penne.data.file_to_stem(dataset, annotation_file)] = [all_total_frames, annotation.shape[1]]
            # update total frames
            all_total_frames += annotation.shape[1]
            # save annotation to numpy
            filename = all_dir / "annotation" / f'{annotation_file.stem}.npy'
            np.save(filename, annotation)
            filename = harmo_dir / "annotation" / f'{annotation_file.stem}.npy'
            np.save(filename, annotation)

            # mask out unvoiced frames
            voiced = (annotation != 0).squeeze()
            annotation = annotation[:,voiced]

            # repeat for voiceonly
            # save offset to start of stem
            voiceonly_offsets[part][penne.data.file_to_stem(dataset, annotation_file)] = [voiceonly_total_frames, annotation.shape[1]]
            # update total frames
            voiceonly_total_frames += annotation.shape[1]
            # save annotation to numpy
            filename = voiceonly_dir / "annotation" / f'{annotation_file.stem}.npy'
            np.save(filename, annotation)


            # PREPROCESS AUDIO

            audio_file = penne.data.stem_to_file(dataset, stem)
            audio, sr = penne.load.audio(audio_file)
            # resample
            audio = penne.resample(audio, sr)
            # save resampled audio to cache
            filename = audio_dir / audio_file.name
            torchaudio.save(filename, audio, penne.SAMPLE_RATE)

            if dataset == 'MDB':
                # pad half windows on ends for MDB
                audio = torch.nn.functional.pad(audio, (penne.WINDOW_SIZE//2, penne.WINDOW_SIZE//2))
            
            # get 1024-sample frames
            frames = torch.nn.functional.unfold(
                    audio[:, None, None, :],
                    kernel_size=(1, penne.WINDOW_SIZE),
                    stride=(1, penne.HOP_SIZE))
            
            # handle off by one error
            if n_annotation_frames < frames.shape[2]:
                frames = frames[:,:,:n_annotation_frames]

            # save all to numpy
            frame_filename = all_dir / "frames" / f'{audio_file.stem}.npy'
            np.save(frame_filename, frames)

            # mask out unvoiced frames
            frames = frames[:,:,voiced]

            # save voiceonly to numpy
            frame_filename = voiceonly_dir / "frames" / f'{audio_file.stem}.npy'
            np.save(frame_filename, frames)

            # save outputted for harmof0
            frames = harmo_instance.pred_file(audio_file)

            frame_filename = harmo_dir / "frames" / f'{audio_file.stem}.npy'
            np.save(frame_filename, frames)

        # store total frames for each partition
        all_offsets["totals"][part] = all_total_frames
        voiceonly_offsets["totals"][part] = voiceonly_total_frames

    # save offsets to json
    with open(all_dir / "offsets.json", 'w') as f:
        json.dump(all_offsets, f, indent=4)
    with open(harmo_dir / "offsets.json", 'w') as f:
        json.dump(all_offsets, f, indent=4)
    with open(voiceonly_dir / "offsets.json", 'w') as f:
        json.dump(voiceonly_offsets, f, indent=4)
