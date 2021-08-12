# This script was produced by the Polavieja Lab. Original source code can be found here:
#   https://gitlab.com/polavieja_lab/fishandra

import numpy as np
import git
import trajectorytools as tt
import os
import logging
import multiprocessing

from .constants import path_to_project_root
from .datasetsimple import DataSet

from .socialcontext import give_me_social_context
from .data_collapse import collapse_data_by_rotating
from . import datasetfrontend
from .shuffle import shuffle_trajectories_in_place, shuffle_social_context_in_place
from . import loader_constants as cons

logger = logging.getLogger(__name__)

def generate_total_config_dict(config_dict):
    ## Generating dictionary from git hash and config values used.
    total_dict = {**config_dict, **cons.cons_dictionary()}
    repo = git.Repo(path_to_project_root, search_parent_directories=True)
    total_dict['githash'] = repo.head.object.hexsha
    logger.debug("Generated dataset total config: {}".format(total_dict))
    return total_dict

def load_many_datasets_from_file(data_files, d):
    # Loading datasets from disk and creating a list
    if cons.NUM_CPU < 0:
        datasets = [load_one_from_file(data_file, d) for data_file in data_files]
    else:
        num_files = len(data_files)
        with multiprocessing.Pool(processes=cons.NUM_CPU) as pool:
            datasets = pool.starmap(load_one_from_file,
                                    zip(data_files, [d]*num_files))

    # Creating a joint dataset using the magic __add__ method
    joint_dataset = datasets[0]
    datasets[0].print_summary()
    for dataset in datasets[1:]:
        dataset.print_summary()
        joint_dataset += dataset
    joint_dataset.print_summary()

    # if 'blind' is not empty, add a frontend
    # This frontend is used to zero certain parameters (e.g. tangential acc)
    if d.get('blind', []):
        logger.warning("blind to {}".format(d['blind']))
        joint_dataset.add_frontend(datasetfrontend.BlindingFrontEnd(d['blind']))

    return joint_dataset

def load_one_from_file(data_file, d):
    """ Convenience function that takes arguments from the dictionary d
    and uses them as keyword arguments when calling "load_dataset_from_file"
    with some sane defaults """
    return load_dataset_from_file(data_file,
                                  num_neighbours=d['num_neighbours'],
                                  future_steps=d['future_steps'],
                                  history_steps=d.get('history_steps', 0),
                                  sigma = d.get('sigma', 1))


def load_dataset_from_file(data_file, **kwargs):
    """ loads a dataset from file. It checks whether there is a preprocessed
    version of the file. If yes, it loads it. If not, it uses "load_dataset"
    to generate it from file
    """

    if data_file.endswith("precalculated.npy"):
        # You cannot load a precalculated file
        # This check will be removed soon
        raise Exception("Loading from file named precalculated not allowed")

    logger.info("Loading " + data_file)
    ## In previous versions, we would be looking here for a precalculated version
    if 'USE_PRECALCULATED' in cons.cons_dictionary():
        logger.warning("USE_PRECALCULATED has no effect now")
    if 'SAVE_PRECALCULATED' in cons.cons_dictionary():
        logger.warning("SAVE_PRECALCULATED has no effect now")

    t_dict = np.load(data_file, encoding='latin1', allow_pickle=True).item()
    tr = t_dict['trajectories']

    # TODO: Code below used to kick in and allow loading from a bare npy
    #    tr = np.load(data_file, allow_pickle=True)
    #    logger.warning("Loading of dict failed. Normal if loading naked trajectories")

    properties = {key: t_dict.get(key, None)
                  for key in ['body_length', 'frames_per_second']}

    new_dataset = load_dataset(tr.astype(np.float32), name=os.path.basename(data_file),
                               properties=properties,
                               data_file = data_file,
                               **kwargs)
    new_dataset.total_config_dict = generate_total_config_dict(kwargs)
    return new_dataset

def one_hot_left_or_right(v,a):
    """ One-hot, whether vector a is to the right or left of vector v
    """
    t = np.sign(np.cross(a,v))
    return np.stack([np.heaviside(t, 0.0), np.heaviside(-t, 0.0)], axis=-1)

def preprocess_trajectories(data, sigma):
    """ Trajectories are smoothed, and acceleration and velocity calculated
    """
    log_fraction_of_nans(data)
    tt.interpolate_nans(data)
    normalization_parameters = tt.center_trajectories_and_normalise(data)
    data_smooth = tt.smooth(data, sigma=sigma, only_past=True)
    trajectories, velocity, acceleration = tt.velocity_acceleration_backwards(data_smooth)
    # There is maybe suffling
    if cons.SHUFFLE == 'trajectories':
        shuffle_trajectories_in_place(trajectories, velocity, acceleration)
    logger.debug("New shapes after preprocessing {} {} {}".format(trajectories.shape, velocity.shape, acceleration.shape))
    return trajectories, velocity, acceleration, normalization_parameters

def log_fraction_of_nans(data):
    """ Calculates and logs the fraction of scenes with nans
    """
    num_nans = np.count_nonzero(np.any(np.isnan(data), axis=(1,2)))
    if num_nans > 0:
        num_scenes = data.shape[1]*data.shape[0]
        logger.warning("NaNs are {0:.3%} of data".format(num_nans/num_scenes))

def cut_all_data(all_data, fractions, datasets_labels):
    """ Cuts all data (trajectories, etc) based on the fractions
    """
    # We calculate the start and end points of each section (training,
    # validation, test, etc).
    num_frames = all_data['asocial'].shape[0]
    cut_indices = list(np.floor(np.cumsum(fractions)*num_frames).astype(np.int))
    start_indices = [0] + cut_indices
    end_indices = cut_indices + [None]

    dataset_list = [DataSet.from_indices(all_data, start_i, end=end_i)
                    for start_i, end_i in zip(start_indices, end_indices)]

    # Dataset dict is the union of all the dataset dicts
    dataset_dict = {}
    for dataset, label in zip(dataset_list, datasets_labels):
        if label in dataset_dict.keys():
            dataset_dict[label] += dataset
        else:
            dataset_dict[label] = dataset
        dataset_dict[label].name = label
    return dataset_dict

def load_dataset(data, future_steps = 5, num_neighbours = 15,
                 history_steps = 0, sigma = 1, name = None,
                 collapse_data = True, properties = {}, data_file = None):

    # Normalize, smooth trajectories and calculate velocity and acceleration
    # Possibly shuffling, depending con cons.SHUFFLE
    trajectories, velocity, acceleration, normalization_parameters = preprocess_trajectories(data, sigma)

    asocial_data = np.concatenate([
        trajectories[:-future_steps], velocity[:-future_steps], acceleration[:-future_steps]], axis=-1)
    # Produce history and cut down trajectories and asocial accordingly
    # This is because we can only use points with full history and
    # with a target
    traj_for_history = trajectories[:-future_steps]
    if history_steps == 0: #Hack, we need a placeholder
        history = None
    elif history_steps > 0:
        history_list = [traj_for_history[i:(-history_steps+i)] for i in
                                         range(history_steps)]
        history_list.append(traj_for_history[history_steps:])
        history = np.stack(history_list, axis=2)
    else:
        raise Exception("History steps cannot be a negative number")

    trajectories = trajectories[history_steps:]
    asocial_data = asocial_data[history_steps:]
    target_trajectories = trajectories[future_steps:] - trajectories[:-future_steps]

    # Create social context: relative positions of neighbours.
    social_context, history_context, neighbours_indices \
        = give_me_social_context(asocial_data, num_neighbours, history=history)
    if cons.SHUFFLE == 'social_context':
        shuffle_social_context_in_place(social_context)

    # Compute target left-right
    left_or_right = one_hot_left_or_right(asocial_data[..., 2:4],
                                          target_trajectories)

    # Save all data arrays into a single dictionary
    all_data = {"asocial": asocial_data, "social": social_context,
                "target": target_trajectories, "turn": left_or_right,
                "neighbours_indices": neighbours_indices}
    if history is not None:
        all_data.update({'history': history_context})

    # We save an uncollapsed copy of the trajectories
    uncollapsed_asocial = asocial_data.copy()
    all_data.update({"uncollapsed": uncollapsed_asocial})

    # Collapse data, i.e. put it into the pseudo-comoving frame of reference
    assert collapse_data
    collapsable = {'asocial': 3, 'social': 3}  # Collapsable pairs
    collapse_data_by_rotating(all_data, collapsable)
    if history is not None:
        collapsable['history'] = 1

    # We tag all points that are too close to the border as invalid
    if cons.REMOVE_OUTER is not None:
        distance_to_center = all_data['asocial'][..., 1]  # Assumes collapsed: position to center of arena is in Y axis.
        all_data["invalid"] = distance_to_center > cons.REMOVE_OUTER

    # We cut the dataset into chunks. Some of them will be training data, some
    # test, some validation...
    dataset_dict = cut_all_data(all_data, cons.FRACTIONS, cons.LABELS)
    logger.info("Keys of dataset_dict {}".format(set(dataset_dict.keys())))

    if 'train' in dataset_dict.keys():
        dynamics_dict = {"mean_speed": np.mean(dataset_dict['train'].data['social'][:,0,3]),
                         "median_speed": np.median(dataset_dict['train'].data['social'][:,0,3]), # 0 is for the focal, 3 is because is vy, and focal animals are oriented in y-axis
                         "max_speed": np.max(dataset_dict['train'].data['social'][:,0,3]), # it should probably be done from the neighbours speeds
                         "speed_percentiles": [np.percentile(dataset_dict['train'].data['social'][:,0,3], per) for per in cons.PERCENTILES],
                         "median_acceleration": np.median(np.linalg.norm(dataset_dict['train'].data['social'][:,0,4:], axis=1)),
                         "max_acceleration": np.median(np.linalg.norm(dataset_dict['train'].data['social'][:,0,4:], axis=1)),
                         "acceleration_percentiles": [np.percentile(np.linalg.norm(dataset_dict['train'].data['social'][:,0,4:], axis=1), per) for per in cons.PERCENTILES]}
    num_data_dict = {'num_' + key: dataset_dict[key].data['social'].shape[0] for key in dataset_dict.keys()}
    config_dict = {"remove_outer": cons.REMOVE_OUTER,
                   "collapsed": collapse_data,
                   "collapsable_pairs": collapsable,
                   "normalization_parameters": normalization_parameters,
                   "percentiles": cons.PERCENTILES,
                   "data_file": data_file,
                   "datasets_labels": cons.LABELS,
                   "fractions": cons.FRACTIONS,
                   "num_frames": data.shape[0],
                   "num_animals": data.shape[1]}
    config_dict = {**config_dict, **dynamics_dict, **num_data_dict}
    properties.update(config_dict)
    return DataSets(dataset_dict, properties, name=name)


class DataSets():
    def __init__(self, dataset_dict, config, name=None):
        """__init__

        :param dataset_dict: A dictionary of DataSet, with keys [validation, test, train]
        :param config: dictionary of config generated when cutting/collapsing DataSet
        :type config: dict
        """
        self.dataset = dataset_dict
        self.config = [config] if not isinstance(config, list) else config
        self.name = name
        logger.debug(config)
        self.link()

    def only_sample_from(self, individuals_to_sample):
        for d in self.dataset:
            d.only_sample_from(individuals_to_sample)

    def set_aliases(self):
        self.validation = self.dataset.get('validation', None)
        self.test = self.dataset.get('test', None)
        self.train = self.dataset.get('train', None)

    def link(self):
        self.set_aliases()
        for key in self.dataset:
            self.dataset[key].parent = self
            self.dataset[key].name = key

    def save(self, filename):
        np.save(filename, self)

    @classmethod
    def load(cls, filename):
        loaded = np.load(filename, allow_pickle=True).item()
        assert(isinstance(loaded, DataSets))
        return loaded

    def print_summary(self):
        for key in self.dataset:
            self.dataset[key].print_summary()

    def add_frontend(self, *args, **kwargs):
        for key in self.dataset:
            self.dataset[key].add_frontend(*args, **kwargs)

    def __add__(self, other):
        #TODO: Make sure that configurations/keys are compatible
        new_dataset = {key: (self.dataset[key] + other.dataset[key]) for key in self.dataset}
        new_config = self.config + other.config
        return DataSets(new_dataset, new_config, name=self.name + "+" + other.name) #### PLZ, handle config in the right way
