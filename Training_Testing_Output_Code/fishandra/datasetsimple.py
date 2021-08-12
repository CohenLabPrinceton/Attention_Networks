# This script was produced by the Polavieja Lab. Original source code can be found here:
#   https://gitlab.com/polavieja_lab/fishandra

import logging
import numpy as np
from .datasetfrontend import NoFrontEnd

logger = logging.getLogger(__name__)

def flatten_first(v):
    if len(v.shape) > 2:
        return v.reshape((v.shape[0]*v.shape[1], ) + v.shape[2:])
    elif len(v.shape) == 2:
        return v.reshape((v.shape[0]*v.shape[1], ))
    else:
        return v

def remove_invalid_frames(data_to_sample):
    assert(data_to_sample['asocial'][...,1].min()>0.0) #Checks collapsed
    assert(0<data_to_sample['asocial'][...,0].max()<1e-7) #Checks collapsed
    num_cases = data_to_sample['asocial'].shape[0]
    invalid_frames = data_to_sample.get("invalid", np.zeros(num_cases, dtype=np.bool))
    if invalid_frames.any():
        num_invalid = np.count_nonzero(invalid_frames)
        logger.debug("{} frames labeled invalid (e.g. close to border) {:.2%}"\
                     .format( num_invalid, num_invalid/invalid_frames.shape[0]))

    for key in ['asocial', 'social', 'target']:
        axis_to_collapse = tuple(range(len(data_to_sample[key].shape)))[1:]
        invalid_frames = np.logical_or(invalid_frames,
                                       np.any(np.isnan(data_to_sample[key]),
                                              axis = axis_to_collapse))
    if invalid_frames.any():
        num_invalid = np.count_nonzero(invalid_frames)
        logger.info("Removing a total of {} frames ({:.2%})".format(
                        num_invalid, num_invalid/invalid_frames.shape[0]))
        for key in data_to_sample:
            data_to_sample[key] = data_to_sample[key][np.where(~invalid_frames)]

def update_with_frame_and_identity_information(all_data):
    """
    Adds (in place) two keys with frame number and identity. This is necessary
    because this information is then lost after collapse and removal
    of frames
    """
    total_time_steps = all_data['social'].shape[0]
    num_individuals = all_data['social'].shape[1]
    ones_array = np.ones((total_time_steps, num_individuals), dtype=np.uint8)
    frames = np.einsum('ij,i->ij', ones_array,
                            np.arange(total_time_steps, dtype=np.uint)
                            )#.reshape((-1))
    all_data.update({"frame_numbers": frames})
    identities = np.einsum('ji,i->ji', ones_array,
                            np.arange(num_individuals, dtype=np.uint8)
                            )#.reshape((-1))
    all_data.update({"identities": identities})

def produce_data_to_sample(data_dict):
    """
    Flattens the first two dimensions, so the first index goes along both
    time and individual. Then, it removes datapoints farther from center than
    TODO: Make it ready for restricted individuals to sample
    """
    data_to_sample = {key: flatten_first(data_dict[key]) for key in data_dict}
    remove_invalid_frames(data_to_sample)
    return data_to_sample

def scramble(data_to_scramble):
    """
    Scrambles all
    """
    total_time_steps = data_to_scramble['social'].shape[0]
    permutation = np.random.permutation(total_time_steps)
    for key in data_to_scramble:
        data_to_scramble[key] = data_to_scramble[key][permutation]

class DataSet():
    def __init__(self, data_to_sample):
        self._name = None
        self.parent = None
        self.data = data_to_sample
        data_to_output = {'social', 'target', 'turn', 'history',
                          'frame_numbers', 'identities', 'neighbours_indices'}
        self.data_to_output = data_to_output.intersection(data_to_sample.keys())
        self._output_dict = dict()
        self.frontend = NoFrontEnd()

    @classmethod
    def from_indices(cls, all_data, start = 0, end = None):
        update_with_frame_and_identity_information(all_data)
        cut_down_data = {key: all_data[key][start:end] for key in all_data}
        data_to_sample = produce_data_to_sample(cut_down_data)
        return cls(data_to_sample)

    def __add__(self,other):
        assert self.num_neighbours == other.num_neighbours
        #assert self.frontend == other.frontend
        data = {key:np.concatenate([self.data[key], other.data[key]],axis=0) for key in self.data}
        dataset = DataSet(data)
        dataset.add_frontend(self.frontend) #WARNING! WATCHOUT FOR FRONTEND MISMATCH
        return dataset

    @property
    def all(self):
        output = {key: self.data[key] for key in self.data_to_output}
        return self.frontend(output)

    @property
    def name(self):
        if self.parent is None:
            return self._name
        else:
            return str(self.parent.name) + " " + str(self._name)

    @name.setter
    def name(self, new_name):
        self._name = new_name

    def add_frontend(self, frontend):
        """Adds a frontend to the dataset.
        I use frontend to make 0 some of the outputs, e.g. blinding the network
        to accelerations or velocities"""
        self.frontend = frontend

    def print_summary(self):
        shape_dict = {key: self.data[key].shape for key in self.data}
        logger.debug("Dataset {} with shapes {}".format(self.name, str(shape_dict)))
        logger.debug("Median speed {}".format(np.median(self.data['social'][...,0,3])))
        logger.debug("Median normal acc. {}".format(np.median(np.abs(self.data['social'][...,0,5]))))
        logger.debug("Fraction turning right {}".format(np.mean(self.data['turn'][...,0])))
        logger.debug("Fraction turning left {}".format(np.mean(self.data['turn'][...,1])))

    @property
    def num_neighbours(self):
        return self.data['social'].shape[1] - 1

    @property
    def num_datapoints(self):
        return self.data['social'].shape[0]
