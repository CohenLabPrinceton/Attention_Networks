# This script was produced by the Polavieja Lab. Original source code can be found here:
#   https://gitlab.com/polavieja_lab/fishandra

import logging

import numpy as np
from trajectorytools.geometry import angle_between_vectors

from .loader import load_many_datasets_from_file
from .model import load_model_from_file
from trajectorytools.geometry import angle_between_vectors

def test(args, logresults=None):
    if logresults is None:
        logresults = logging.getLogger(__name__)

    # Loading best model and outputting statistics
    model = load_model_from_file(args.model_folder)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.args)

    # Create the data loader object. It preprocess the data in terms of
    # batches each of size args.batch_size, of length args.seq_length
    data_files = args.data
    print("Loading ", data_files)

    ## Attention! Using loaded config from model
    datasets = load_many_datasets_from_file(data_files, model.args)

    labels = {'train': datasets.train.all['turn'],
              'test': datasets.test.all['turn'],
              'validation': datasets.validation.all['turn']}
    data = {'train': datasets.train.all['social'],
            'test': datasets.test.all['social'],
            'validation': datasets.validation.all['social']}


    val_loss, val_acc = model.evaluate(data['validation'],
                                       labels['validation'])
    print('Best validation Acc:', val_acc)
    test_loss, test_acc = model.evaluate(data['test'], labels['test'])
    print('Best test Acc:', test_acc)

    test_real_angle_rads = angle_between_vectors(np.array([0,1]),
                                    datasets.test.all['target']).flatten()
    test_real_angle = np.degrees(test_real_angle_rads)
    large_angles = np.where((test_real_angle > 20) & (test_real_angle < 160))
    top_angles = np.where((test_real_angle > 30) & (test_real_angle < 100))
    test_loss_la, test_acc_la = model.evaluate(data['test'][large_angles],
                                               labels['test'][large_angles])
    test_loss_ta, test_acc_ta = model.evaluate(data['test'][top_angles],
                                               labels['test'][top_angles])

    logresults.info("Best: test loss {} validation loss {}"
                    .format(test_loss, val_loss))
    logresults.info("Best: accuracy {} for large turns {}, for top turns {}"
                    .format(test_acc, test_acc_la, test_acc_ta))

    vars_to_save = ['val_loss', 'val_acc',
                    'test_loss', 'test_loss_la', 'test_loss_ta',
                    'test_acc', 'test_acc_la', 'test_acc_ta']
    test_dict = {}
    for var in vars_to_save:
        test_dict[var] = locals()[var]
    test_dict['num_validation'] = len(labels['validation'])
    test_dict['num_test'] = len(labels['test'])
    test_dict['num_test_la'] = len(labels['test'][large_angles])
    test_dict['num_test_ta'] = len(labels['test'][top_angles])
    return test_dict
