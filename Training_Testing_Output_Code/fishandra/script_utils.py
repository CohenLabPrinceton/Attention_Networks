import argparse
import os

from fishandra.plot.new_constants import constants as cons
from fishandra.model import load_model_from_file

def load_model_and_update_constants(args, one_neighbour=False):
    data_config_path = os.path.join(args.model_folder, 'dataset_config.npy')
    cons.update_constants(data_config_path,
                          fixed_speeds=not args.not_fixed_speeds,
                          fixed_units=not args.not_fixed_units)
    if one_neighbour:
        model = load_model_from_file(args.model_folder,
                                     updated_args={'num_neighbours': 1})
    else:
        model = load_model_from_file(args.model_folder)
    return model

def generate_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_folder', type=str, help='Path to the folder where\
                        the model.h5, model_config.npy and data_config.npy are')
    parser.add_argument('--not_save_results', action='store_true')
    parser.add_argument('--not_fixed_speeds', action='store_true')
    parser.add_argument('--not_fixed_units', action='store_true')
    return parser

