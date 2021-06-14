import argparse
import json
import os

import numpy as np
import pescador
import shared
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import train
from tqdm import tqdm

TEST_BATCH_SIZE = 64


def prediction(batch_dispatcher, tf_vars):
    pred_list, id_list = [], []

    [sess, normalized_y, cost, x, y_, is_train] = tf_vars
    for batch in tqdm(batch_dispatcher):
        pred, _ = sess.run([normalized_y, cost], feed_dict={x: batch['X'],
                                                            y_: batch['Y'],
                                                            is_train: False})

        id_list.append(batch['ID'])
        pred_list.append(pred)

    id_array = np.hstack(id_list)
    pred_array = np.vstack(pred_list)

        if not array_cost:  # if array_cost is empty, is the first iteration
            pred_array = pred
            id_array = batch['ID']
        else:
            pred_array = np.concatenate((pred_array,pred), axis=0)
            id_array = np.append(id_array,batch['ID'])
        array_cost.append(cost_pred)
    print('predictions', pred_array.shape)
    return pred_array, id_array


if __name__ == '__main__':
    # which experiment we want to evaluate?
    # Use the -l functionality to ensamble models: python arg.py -l 1234 2345 3456 4567
    parser = argparse.ArgumentParser()
    parser.add_argument('index_file')
    parser.add_argument('groundtruth_file')
    parser.add_argument('model_dir')
    parser.add_argument('data_dir')
    parser.add_argument('predictions_file')
    parser.add_argument('-l', '--list', nargs='+', help='List of models to evaluate', required=True)
    parser.add_argument(
        '--data-dirs',
        nargs='+',
        help=(
            'a list of directories containing feature (.dat) files. '
            'The first one would be consdired the main feature. '
            'This list should contain the same number of elements as `--feature-type`.'
        )
    )

    args = parser.parse_args()
    models = args.list
    index_file = args.index_file
    groundtruth_file = args.groundtruth_file
    model_dir = args.model_dir
    data_dir = args.data_dir
    predictions_file = args.predictions_file

    feature_combination = data_dir == 'multiple_directories'

    if feature_combination:
        from data_loaders import data_gen_feature_combination as data_gen
        data_dirs = args.data_dirs
        data_dir = data_dirs[0]
    else:
        from data_loaders import data_gen_standard as data_gen

    # load all audio representation paths
    [_, id2audio_repr_path] = shared.load_id2path(index_file)
    id2audio_repr_path = {k: v for k, v in id2audio_repr_path.items()}

    index_ids = set(id2audio_repr_path.keys())

    gt_ids = set(json.load(open(groundtruth_file, 'r')).keys())

    ids = list(index_ids.intersection(gt_ids))

    print('{} ids found in the index file'.format(len(index_ids)))
    print('{} ids found in the groundtruth file'.format(len(gt_ids)))
    print('using {} intersecting ids'.format(len(ids)))

    for model in models:
        experiment_folder = os.path.join(model_dir, str(model))
        config = json.load(open(os.path.join(experiment_folder, 'config.json')))

        if feature_combination:
            config['audio_representation_dirs'] = data_dirs
        else:
            config['audio_representation_dir'] = data_dir

        print('Experiment: ' + str(model))
        print('\n' + str(config))

        # pescador: define (finite, batched & parallel) streamer
        pack = [config, 'overlap_sampling', config['xInput']]
        streams = [pescador.Streamer(data_gen, id, id2audio_repr_path[id], [0] * config['num_classes_dataset'], pack) for id in ids]
        mux_stream = pescador.ChainMux(streams, mode='exhaustive')
        batch_streamer = pescador.Streamer(pescador.buffer_stream, mux_stream, buffer_size=TEST_BATCH_SIZE, partial=True)
        batch_streamer = pescador.ZMQStreamer(batch_streamer)

        # tensorflow: define model and cost
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()

            [x, y_, is_train, y, normalized_y, cost, model_vars] = train.tf_define_model_and_cost(config)
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            results_folder = experiment_folder + '/'
            saver.restore(sess, results_folder)
            tf_vars = [sess, normalized_y, cost, x, y_, is_train]

            pred_array, id_array = prediction(batch_streamer, tf_vars)
            sess.close()

    print('Predictions computed, now evaluating..')

    y_pred, n_ids = shared.average_predictions_ids(pred_array, id_array, ids)

    print('len y_pred: ', len(y_pred))
    print('len n_ids: ', len(n_ids))
    predictions = {id: list(pred.astype('float64')) for id, pred in zip(n_ids, y_pred)}

    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)
