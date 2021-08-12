# This script was produced by the Polavieja Lab. Original source code can be found here:
#   https://gitlab.com/polavieja_lab/fishandra

import logging
import os
import time
from datetime import datetime

import numpy as np
from keras import callbacks
from trajectorytools.geometry import angle_between_vectors

from . import constants
from .loader import load_many_datasets_from_file
from . import loader_constants as loader_constants
from .model import load_model, load_model_from_file
from .save_train_results import save_train_results


def create_folder(save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)


def train(args, logresults=None, save_results=True):
    start_train = time.time()
    create_folder(args.save_path)

    if logresults is None:
        logresults = logging.getLogger(__name__)

    constants_to_print = {key: loader_constants.__dict__[key]
                          for key in loader_constants.__dict__
                          if not key.startswith('_')}

    if constants.DRY_RUN:
        logresults.warning("RUNNING DRY! -- no training is performed")

    logresults.info("Starting training at {}".format(
                                datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    logresults.info("Options were: {}".format(args))
    logresults.info("Constants were: {}".format(constants_to_print))

    # Create the data loader object. It preprocess the data in terms of
    # batches each of size args.batch_size, of length args.seq_length
    data_files = args.data
    logresults.info("Loading {} files".format(len(data_files)))
    loader_dict = {k: vars(args)[k] for k in vars(args).keys() &
                   {'num_neighbours', 'future_steps', 'topological_attention',
                    'sigma', 'blind', 'model', 'history_steps', 'attention_variables',
                    'integration_arch', 'pairwise_arch', 'attention_arch'}}

    datasets = load_many_datasets_from_file(data_files, loader_dict)
    labels, data = {}, {}
    for key in ['train', 'test', 'validation']:
        labels[key] = datasets.dataset[key].all['turn']
        if args.model == 'attention_history':
            assert args.history_steps > 0
            data[key] = [datasets.dataset[key].all['social'],
                         datasets.dataset[key].all['history']]
        else:
            data[key] = datasets.dataset[key].all['social']

    interaction = load_model(loader_dict)
    interaction.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    interaction.summary()

    if constants.DRY_RUN:
        print("Shapes of labels {}".format({key: labels[key].shape
                                            for key in labels}))
        try:
            print("Shapes of data {}".format({key: data[key].shape
                                            for key in data}))
        except: #Sometimes data is a list of arrays, e.g. (social, history)
            print("Shapes of data {}".format({key: [a.shape for a in data[key]]
                                            for key in data}))
        end_train = time.time()
        logresults.info("Dry training finished. It took {} seconds!".
                        format(end_train-start_train))
        return

    # Training
    start_lr = args.learning_rate
    end_lr = args.min_learning_rate
    decay_lr = args.decay_rate

    def schedule(epoch):
        return end_lr + (start_lr - end_lr)*decay_lr**(epoch)

    callback_list = [callbacks.LearningRateScheduler(schedule, verbose=1),
                    callbacks.ModelCheckpoint(os.path.join(args.save_path, 'model.h5'),
                                              monitor='val_loss',
                                              verbose=1, save_best_only=True,
                                              save_weights_only=True,
                                              mode='auto', period=1),
                    callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                            patience=45, verbose=1,
                                            baseline=None)
                    ]


    interaction.save(os.path.join(args.save_path, ))
    history = interaction.fit(
        data['train'],
        labels['train'],
        callbacks=callback_list,
        validation_data=(
            data['validation'],
            labels['validation']),
        epochs=args.num_epochs,
        batch_size=args.batch_size)
    test_loss, test_acc = interaction.evaluate(data['test'], labels['test'])
    print('test_acc:', test_acc)

    # Loading best model and outputting statistics
    model = load_model_from_file(args.save_path)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    val_loss, val_acc = model.evaluate(data['validation'],
                                       labels['validation'])
    print('Best validation Acc:', val_acc)
    test_loss, test_acc = model.evaluate(data['test'], labels['test'])
    print('Best test Acc:', test_acc)

    test_real_angle_rads = angle_between_vectors(
        np.array([0, 1]), datasets.test.all['target']).flatten()
    test_real_angle = np.degrees(test_real_angle_rads)
    large_angles = np.where((test_real_angle > 20) & (test_real_angle < 160))
    top_angles = np.where((test_real_angle > 30) & (test_real_angle < 100))

    if large_angles[0].size == 0:
        print(large_angles)
        logresults.warning("No large angles")
        test_loss_la, test_acc_la = np.nan, np.nan
    else:
        if args.model == 'attention_history': #TODO: General
            data_test_large_angles = [data['test'][0][large_angles],
                                      data['test'][1][large_angles]]
        else:
            data_test_large_angles = data['test'][large_angles]

        test_loss_la, test_acc_la = model.evaluate(
            data_test_large_angles, labels['test'][large_angles])

    if top_angles[0].size == 0:
        print(top_angles)
        logresults.warning("No top angles")
        test_loss_ta, test_acc_ta = np.nan, np.nan
    else:
        if args.model == 'attention_history': #TODO: General
            data_test_top_angles = [data['test'][0][top_angles],
                                      data['test'][1][top_angles]]
        else:
            data_test_top_angles = data['test'][top_angles]

        test_loss_ta, test_acc_ta = model.evaluate(data_test_top_angles,
                                                   labels['test'][top_angles])

    logresults.info("Best: test loss {} validation loss {}"
                    .format(test_loss, val_loss))
    logresults.info("Best: accuracy {} for large turns {}, for top turns {}"
                    .format(test_acc, test_acc_la, test_acc_ta))

    end_train = time.time()
    logresults.info("Training finished. It took {} seconds!".
                    format(end_train-start_train))

    if save_results:
        np.save(os.path.join(args.save_path, 'dataset_config.npy'),
                datasets.config)
        save_train_results(locals())
