import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from feature_melspectrogram import MelSpectrogramMusiCNN, MelSpectrogramVGGish


def compute_audio_repr(audio_file, audio_repr_file, extractor, force=False):
    if not force:
        if audio_repr_file.exists():
            print('{} exists. skipping!'.format(audio_repr_file))
            return 0

    audio_repr = extractor.compute(audio_file)

    # Compute length
    length = audio_repr.shape[0]

    # Transform to float16 (to save storage, and works the same)
    audio_repr = audio_repr.astype(np.float16)

    # Write results:
    fp = np.memmap(audio_repr_file, dtype='float16', mode='w+', shape=audio_repr.shape)
    fp[:] = audio_repr[:]
    del fp
    return length


def do_process(files, index, extractor, audio_representation_dir):
    try:
        [id, audio_file, audio_repr_file] = files[index]

        audio_repr_file = Path(audio_repr_file)
        audio_repr_file.parent.mkdir(parents=True, exist_ok=True)
        # compute audio representation (pre-processing)
        compute_audio_repr(audio_file, audio_repr_file, extractor)
        # index.tsv writing
        fw = open(audio_representation_dir / "index.tsv", "a")
        fw.write("%s\t%s\n" % (id, audio_repr_file.relative_to(audio_representation_dir)))
        fw.close()
        # print(str(index) + '/' + str(len(files)) + ' Computed: %s' % audio_file)

    except Exception as e:
        ferrors = open(audio_representation_dir / "errors.txt", "a")
        ferrors.write(audio_file + "\n")
        ferrors.write(str(e))
        ferrors.close()
        print('Error computing audio representation: ', audio_file)
        print(str(e))


def process_files(files, audio_representation_dir, feature_type=None, config=None):

    assert feature_type or config, "At least one should be provided."

    # it not provided explicitly read it from the config
    if not feature_type:
        feature_type = config['config_train']['feature_type']

    if feature_type == 'waveform':
        extractor = None
    elif feature_type == 'musicnn-melspectrogram':
        extractor = MelSpectrogramMusiCNN()
    elif feature_type == 'vggish-melspectrogram':
        extractor = MelSpectrogramVGGish()

    # import only the feature extractors that we need. This is because for the spectrogram
    # features `essentia` but the embeddings require `essentia-tensorflow`
    elif feature_type in ('effnet_b0', 'effnet_b0_3M', 'musicnn', 'openl3', 'tempocnn', 'vggish', 'yamnet', 'maest'):
        from feature_embeddings import EmbeddingFromMelSpectrogram
        extractor = EmbeddingFromMelSpectrogram(feature_type)

    elif feature_type == 'spleeter':
        from feature_embeddings import EmbeddingFromWaveform
        extractor = EmbeddingFromWaveform(feature_type)

    elif feature_type == 'jukebox':
        from feature_jukebox import Jukebox
        extractor = Jukebox()

    else:
        raise NotImplementedError('Feature {} not implemented.'.format(feature_type))

    for index in tqdm(range(0, len(files))):
        do_process(files, index, extractor, audio_representation_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='configuration file')
    args = parser.parse_args()

    config = json.load(open(args.config_file, "r"))

    audio_representation_dir = Path(config['config_train']['audio_representation_dir'])

    # set audio representations folder
    audio_representation_dir.mkdir(parents=True, exist_ok=True)

    # list audios to process: according to 'index_file'
    files_to_convert = []
    f = open(Path(config['data_dir'], config['config_preprocess']['index_audio_file']))
    for line in f.readlines():
        id, audio = line.strip().split("\t")
        audio_repr = audio[:audio.rfind(".")] + ".dat" # .npy or .pk
        files_to_convert.append((id, str(Path(config['config_preprocess']['audio_dir'], audio)),
                                 str(audio_representation_dir / audio_repr)))

    # compute audio representation
    process_files(files_to_convert, audio_representation_dir, config=config)

    print("Audio representation folder: ", audio_representation_dir)
