import numpy as np
import random
import os
from pathlib import Path


def compress(audio_rep, compression=None):
    # do not apply any compression to the embeddings
    if not compression:
        return audio_rep
    elif compression == 'logEPS':
        return np.log10(audio_rep + np.finfo(float).eps)
    elif compression == 'logC':
        return np.log10(10000 * audio_rep + 1)
    else:
        raise('get_audio_rep: Preprocessing not available.')


def get_short_rep(audio_repr_path, x, y, frames_num):
    fp = np.memmap(audio_repr_path, dtype='float16',
                   mode='r', shape=(frames_num, y))
    audio_rep = np.zeros([x, y])
    audio_rep[:frames_num, :] = np.array(fp)
    del fp

    return audio_rep


def read_mmap(audio_repr_path, x, y, frames_num, single_patch=False, offset=0, compression=None):
    if frames_num < x:
        audio_repr = get_short_rep(audio_repr_path, x, y, frames_num)
    else:
        read_x = x if single_patch else frames_num
        fp = np.memmap(audio_repr_path, dtype='float16',
                       mode='r', shape=(read_x, y), offset=offset)
        audio_repr = np.array(fp)
        del fp
    return compress(audio_repr, compression=compression)


def data_gen_standard(id, audio_repr_path, gt, pack):
    # Support both the absolute and relative path input cases
    config, sampling, param_sampling = pack
    audio_repr_path = Path(config['audio_representation_dir'], audio_repr_path)

    try:
        floats_num = os.path.getsize(audio_repr_path) // 2  # each float16 has 2 bytes
        frames_num = floats_num // config['yInput']

        # let's deliver some data!
        if sampling == 'random':
            for i in range(0, param_sampling):
                random_frame_offset = random.randint(
                    0, frames_num - config['xInput'])
                # idx * bands * bytes per float
                offset = random_frame_offset * config['yInput'] * 2
                yield {
                    'X': read_mmap(audio_repr_path,
                                   config['xInput'],
                                   config['yInput'],
                                   frames_num,
                                   single_patch=True,
                                   offset=offset,
                                   compression=config['feature_params']['compression']
                                   ),
                    'Y': gt,
                    'ID': id
                }

        elif sampling == 'overlap_sampling':
            audio_rep = read_mmap(audio_repr_path,
                                  config['xInput'],
                                  config['yInput'],
                                  frames_num,
                                  compression=config['feature_params']['compression']
                                  )
            last_frame = int(audio_rep.shape[0]) - int(config['xInput']) + 1
            for time_stamp in range(0, last_frame, param_sampling):
                yield {
                    'X': audio_rep[time_stamp: time_stamp + config['xInput'], :],
                    'Y': gt,
                    'ID': id
                }
    except Exception as ex:
        print('"{}" failed'.format(audio_repr_path))
        print(repr(ex))


def data_gen_feature_combination(id, audio_repr_path, gt, pack):
    # Support both the absolute and relative path input cases
    config, sampling, param_sampling = pack
    audio_repr_paths = [Path(p, audio_repr_path)
                        for p in config['audio_representation_dirs']]

    yInputs = [config['features_params'][i]['yInput']
               for i in range(len(config['features_params']))]

    try:
        float_nums = [path.stat().st_size // 2 for path in audio_repr_paths]

        # get the number of frames for each represention. Ideally they should be identical,
        # but different analysis parameters may result in slight differences.
        frames_nums = np.array([n // yInputs[i] for i, n in enumerate(float_nums)])

        frames_range = frames_nums.max() - frames_nums.min()
        assert frames_range < 10, ('The number of frames for at least one of the features '
                                   f'is too diverging: {frames_nums}')

        # use the shortest feature as reference
        frames_num = min(frames_nums)

        # let's deliver some data!
        if sampling == 'random':
            for i in range(0, param_sampling):
                random_frame_offset = random.randint(
                    0, frames_num - config['xInput'])

                x = np.hstack([read_mmap(path,
                                         config['xInput'],
                                         yInputs[i],
                                         frames_num,
                                         single_patch=True,
                                         offset=random_frame_offset * yInputs[i] * 2,
                                         compression=config['feature_params']['compression']
                                         ) for i, path in enumerate(audio_repr_paths)]
                              )
                yield {
                    'X': x,
                    'Y': gt,
                    'ID': id
                }

        elif sampling == 'overlap_sampling':
            x = np.hstack([read_mmap(path,
                                     config['xInput'],
                                     yInputs[i],
                                     frames_num,
                                     compression=config['feature_params']['compression']
                                     ) for i, path in enumerate(audio_repr_paths)]
                          )
            last_frame = int(x.shape[0]) - int(config['xInput']) + 1
            for time_stamp in range(0, last_frame, param_sampling):
                yield {
                    'X': x[time_stamp: time_stamp + config['xInput'], :],
                    'Y': gt,
                    'ID': id
                }
    except Exception as ex:
        print('"{}" failed'.format(audio_repr_path))
        print(repr(ex))
