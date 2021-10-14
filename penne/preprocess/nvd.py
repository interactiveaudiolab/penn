import argparse
import penne
import itertools
from penne.load import pitch_annotation
import torch
from pathlib import Path

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        help='The name of the dataset to preprocess')
    parser.add_argument(
        'checkpoint',
        help='The path to the checkpoint to use for preprocessing')
    parser.add_argument(
        'gpu',
        type=int,
        help='The gpu to use for preprocessing'
    )
    return parser.parse_args()

def from_dataset(dataset, checkpoint, gpu):
    stem_dict = penne.data.partitions(dataset)
    stems = list(itertools.chain(*list(stem_dict.values())))

    for stem in stems:
        audio_file = penne.data.stem_to_file('MDB', stem)
        audio, sr = penne.load.audio(audio_file)
        # resample
        audio = penne.resample(audio, sr)
        results = []

        # Postprocessing breaks gradients, so just don't compute them
        with torch.no_grad():

            # Preprocess audio
            generator = penne.preprocess_from_audio(audio,
                                penne.SAMPLE_RATE,
                                penne.HOP_SIZE,
                                1024,
                                'cpu' if gpu is None else f'cuda:{gpu}',
                                dataset != 'PTDB')
            for frames in generator:

                # Infer independent probabilities for each pitch bin
                logits = penne.infer(frames, Path(checkpoint))

                # Place on same device as audio to allow very long inputs
                logits = logits.to(audio.device)

                results.append(logits)

        # Concatenate
        # double check dimensions (360, x)
        torch.save(torch.cat(results, 0).T, penne.CACHE_DIR / 'nvd' / dataset / 'logits' / f'{stem}.pt')

        # concatenate log2(annotations) and voicings and save (2,length)
        annotation, voicing = pitch_annotation(dataset, penne.data.stem_to_annotation(dataset, stem), False)
        torch.save(torch.cat((torch.log2(annotation), voicing)), penne.CACHE_DIR / 'nvd' / dataset / 'targets' / f'{stem}.pt')


if __name__ == '__main__':
    from_dataset(**vars(parse_args()))