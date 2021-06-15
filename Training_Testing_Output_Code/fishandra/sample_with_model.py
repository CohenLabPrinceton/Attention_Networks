import os
import numpy as np
import logging

from . import loader_constants as cons
from .plot.utils import load_model_and_data
logger = logging.getLogger(__name__)

def sample_with_model(model_folder, max_batch_size=2500, save=True):
    logger.info("Sampling for model {}...".format(model_folder))

    model, datasets = load_model_and_data(model_folder,
                                          loader_constants_dict={'FRACTIONS': (),
                                                                 'LABELS': ['train']},
                                          new_constants_dict={'fixed_units': True,
                                                              'fixed_speeds': True})

    dataset = datasets.train.all
    bs = min(datasets.train.all['social'].shape[0], cons.MAX_BATCH_SIZE)
    logger.info("Predicting attention layer...")
    dataset["attention_layer"] = model.attention_layer.predict(dataset['social'], batch_size=bs)
    logger.info("Predicting...")
    dataset["prediction"] = model.predict(dataset['social'], batch_size=bs)

    if save:
        outfile = os.path.join(model_folder, 'output_results.npy')
        np.save(outfile, dataset)
    return dataset #s

### temporary
def sample_with_model_separate(model_folder, max_batch_size=2500, save=True):
    logger.info("Sampling for model {}...".format(model_folder))
    i=0
    while(True):
        try:
            model, datasets = load_model_and_data(model_folder, datafile_index=i,
                                loader_constants_dict={'REMOVE_OUTER': 1.0,
                                                       'FRACTIONS': (),
                                                       'LABELS': ['train']},
                                new_constants_dict={'fixed_units': True,
                                                    'fixed_speeds': True})
        except IndexError:
            break

        dataset = datasets.train.all
        bs = min(datasets.train.all['social'].shape[0], cons.MAX_BATCH_SIZE)
        logger.info("Predicting attention layer...")
        dataset["attention_layer"] = model.attention_layer.predict(dataset['social'], batch_size=bs)
        logger.info("Predicting...")
        dataset["prediction"] = model.predict(dataset['social'], batch_size=bs)

        if save:
            outfile = os.path.join(model_folder,
                                   'output_results_{}.npy'.format(i))
            np.save(outfile, dataset)
        i += 1
    return dataset #s


